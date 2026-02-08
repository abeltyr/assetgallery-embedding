/**
 * LangGraph embedding pipeline.
 *
 * Graph nodes:
 *   1. readFromSqlite  — reads a batch of rows from SQLite
 *   2. parseAndBuildText — parses raw_json, builds combined text
 *   3. generateEmbeddings — calls the embedding provider
 *   4. saveToPostgres — upserts section data + embedding to PG
 *
 * The graph runs sequentially: read -> parse -> embed -> save
 */

import { StateGraph, Annotation, END, START } from "@langchain/langgraph";
import type { EmbeddingProvider, SqliteRow, RawJsonParsed } from "../types/index.js";
import { SqliteReader } from "../modules/sqlite-reader.js";
import { PgWriter } from "../modules/pg-writer.js";
import { parseRawJson, buildCombinedText } from "../modules/text-builder.js";

// ---------------------------------------------------------------------------
// State definition using LangGraph Annotation
// ---------------------------------------------------------------------------
const PipelineAnnotation = Annotation.Root({
  rows: Annotation<SqliteRow[]>({ reducer: (_, b) => b, default: () => [] }),
  parsedJsons: Annotation<RawJsonParsed[]>({ reducer: (_, b) => b, default: () => [] }),
  combinedTexts: Annotation<string[]>({ reducer: (_, b) => b, default: () => [] }),
  embeddings: Annotation<number[][]>({ reducer: (_, b) => b, default: () => [] }),
  processedCount: Annotation<number>({ reducer: (a, b) => a + b, default: () => 0 }),
  errorCount: Annotation<number>({ reducer: (a, b) => a + b, default: () => 0 }),
  errors: Annotation<Array<{ sectionId: string; error: string }>>({
    reducer: (a, b) => [...a, ...b],
    default: () => [],
  }),
  // Control flags
  batchOffset: Annotation<number>({ reducer: (_, b) => b, default: () => 0 }),
  batchSize: Annotation<number>({ reducer: (_, b) => b, default: () => 50 }),
  totalValid: Annotation<number>({ reducer: (_, b) => b, default: () => 0 }),
  done: Annotation<boolean>({ reducer: (_, b) => b, default: () => false }),
});

type PipelineState = typeof PipelineAnnotation.State;

// ---------------------------------------------------------------------------
// Build the graph
// ---------------------------------------------------------------------------
export function createEmbeddingPipeline(
  sqliteReader: SqliteReader,
  pgWriter: PgWriter,
  embeddingProvider: EmbeddingProvider,
) {
  // --- Node: Read batch from SQLite ---
  async function readFromSqlite(state: PipelineState): Promise<Partial<PipelineState>> {
    const rows = sqliteReader.readBatch(state.batchOffset, state.batchSize);
    console.log(
      `[read] Batch at offset ${state.batchOffset}: got ${rows.length} rows`,
    );

    if (rows.length === 0) {
      return { rows: [], done: true };
    }

    return { rows, done: false };
  }

  // --- Node: Parse JSON and build combined text ---
  async function parseAndBuildText(state: PipelineState): Promise<Partial<PipelineState>> {
    if (state.done || state.rows.length === 0) {
      return { parsedJsons: [], combinedTexts: [] };
    }

    const parsedJsons: RawJsonParsed[] = [];
    const combinedTexts: string[] = [];
    const validRows: SqliteRow[] = [];
    const errors: Array<{ sectionId: string; error: string }> = [];

    for (const row of state.rows) {
      try {
        const parsed = parseRawJson(row.raw_json);
        const text = buildCombinedText(parsed);
        parsedJsons.push(parsed);
        combinedTexts.push(text);
        validRows.push(row);
        console.log("text", text)
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        errors.push({ sectionId: row.section_id, error: `Parse error: ${msg}` });
        console.warn(`[parse] Failed for ${row.section_id}: ${msg}`);
      }
    }

    console.log(
      `[parse] Parsed ${parsedJsons.length} rows, ${errors.length} errors`,
    );

    return {
      rows: validRows,
      parsedJsons,
      combinedTexts,
      errorCount: errors.length,
      errors,
    };
  }

  // --- Node: Generate embeddings ---
  async function generateEmbeddings(state: PipelineState): Promise<Partial<PipelineState>> {
    if (state.done || state.combinedTexts.length === 0) {
      return { embeddings: [] };
    }

    console.log(
      `[embed] Generating embeddings for ${state.combinedTexts.length} texts with ${embeddingProvider.name}/${embeddingProvider.model}`,
    );

    try {
      const embeddings = await embeddingProvider.embedBatch(state.combinedTexts);
      console.log(`[embed] Generated ${embeddings.length} embeddings`);
      return { embeddings };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`[embed] Batch embedding failed: ${msg}`);
      // Mark all rows as errors
      const errors = state.rows.map((r) => ({
        sectionId: r.section_id,
        error: `Embedding error: ${msg}`,
      }));
      return {
        embeddings: [],
        errorCount: state.rows.length,
        errors,
      };
    }
  }

  // --- Node: Save to PostgreSQL ---
  async function saveToPostgres(state: PipelineState): Promise<Partial<PipelineState>> {
    if (state.done || state.embeddings.length === 0) {
      return {};
    }

    console.log(`[save] Upserting ${state.rows.length} sections to PostgreSQL`);

    const result = await pgWriter.upsertBatch(
      state.rows,
      state.parsedJsons,
      state.combinedTexts,
      state.embeddings,
      embeddingProvider.name,
      embeddingProvider.model,
    );

    console.log(
      `[save] Saved ${result.success}, errors: ${result.errors.length}`,
    );

    return {
      processedCount: result.success,
      errorCount: result.errors.length,
      errors: result.errors,
      // Advance the offset for next batch
      batchOffset: state.batchOffset + state.batchSize,
    };
  }

  // --- Conditional edge: continue or stop ---
  function shouldContinue(state: PipelineState): "readFromSqlite" | typeof END {
    if (state.done) return END;
    if (state.batchOffset >= state.totalValid) return END;
    return "readFromSqlite";
  }

  // --- Build the graph ---
  const graph = new StateGraph(PipelineAnnotation)
    .addNode("readFromSqlite", readFromSqlite)
    .addNode("parseAndBuildText", parseAndBuildText)
    .addNode("generateEmbeddings", generateEmbeddings)
    .addNode("saveToPostgres", saveToPostgres)
    // Edges
    .addEdge(START, "readFromSqlite")
    .addEdge("readFromSqlite", "parseAndBuildText")
    .addEdge("parseAndBuildText", "generateEmbeddings")
    .addEdge("generateEmbeddings", "saveToPostgres")
    .addConditionalEdges("saveToPostgres", shouldContinue);

  return graph.compile();
}
