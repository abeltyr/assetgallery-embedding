/**
 * LangGraph embedding pipeline.
 *
 * Graph nodes:
 *   1. readFromSqlite         — reads a batch of rows from SQLite
 *   2. filterAlreadyProcessed — removes rows already saved in PG
 *   3. parseAndBuildText      — parses raw_json, builds combined text
 *   4. generateEmbeddings     — calls the embedding provider (or reads from local cache)
 *   5. cacheEmbeddings        — persists embeddings to local JSONL BEFORE any PG write
 *   6. saveToPostgres         — upserts section data + embedding to PG
 *
 * The cache guarantees: once an embedding is generated, it is on disk.
 * If PG save fails, --retry can replay from cache without re-calling the API.
 */

import { StateGraph, Annotation, END, START } from "@langchain/langgraph";
import type { EmbeddingProvider, SqliteRow, RawJsonParsed } from "../types/index.js";
import { SqliteReader } from "../modules/sqlite-reader.js";
import { PgWriter } from "../modules/pg-writer.js";
import { EmbeddingCache } from "../modules/embedding-cache.js";
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
  skippedCount: Annotation<number>({ reducer: (a, b) => a + b, default: () => 0 }),
  cachedCount: Annotation<number>({ reducer: (a, b) => a + b, default: () => 0 }),
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
  cache: EmbeddingCache,
) {
  // --- Node 1: Read batch from SQLite ---
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

  // --- Node 2: Filter out sections already fully saved in PG ---
  async function filterAlreadyProcessed(state: PipelineState): Promise<Partial<PipelineState>> {
    if (state.done || state.rows.length === 0) {
      return { rows: [], skippedCount: 0 };
    }

    const sectionIds = state.rows.map((r) => r.section_id);
    const alreadyInPg = await pgWriter.getAlreadyProcessedIds(sectionIds);

    if (alreadyInPg.size === 0) {
      console.log(`[filter] No duplicates — all ${state.rows.length} rows are new`);
      return { skippedCount: 0 };
    }

    const newRows = state.rows.filter((r) => !alreadyInPg.has(r.section_id));
    const skipped = state.rows.length - newRows.length;

    console.log(
      `[filter] Skipped ${skipped} already-processed sections, ${newRows.length} remaining`,
    );

    return { rows: newRows, skippedCount: skipped };
  }

  // --- Node 3: Parse JSON and build combined text ---
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

  // --- Node 4: Generate embeddings (or use cached) ---
  async function generateEmbeddings(state: PipelineState): Promise<Partial<PipelineState>> {
    if (state.done || state.combinedTexts.length === 0) {
      return { embeddings: [] };
    }

    // Split rows into cached vs uncached
    const cachedIndices: number[] = [];
    const uncachedIndices: number[] = [];

    for (let i = 0; i < state.rows.length; i++) {
      if (cache.has(state.rows[i].section_id)) {
        cachedIndices.push(i);
      } else {
        uncachedIndices.push(i);
      }
    }

    // Pre-fill embeddings array
    const embeddings: (number[] | null)[] = new Array(state.rows.length).fill(null);

    // For cached: read from cache
    if (cachedIndices.length > 0) {
      const allSavedIds = await pgWriter.getAlreadyProcessedIds([]);
      const unsaved = cache.getUnsavedEntries(allSavedIds);
      const unsavedMap = new Map(unsaved.map((e) => [e.section_id, e.embedding]));

      for (const i of cachedIndices) {
        const cached = unsavedMap.get(state.rows[i].section_id);
        if (cached) {
          embeddings[i] = cached;
        } else {
          // Cached but also already in PG — shouldn't happen due to filter, but be safe
          uncachedIndices.push(i);
        }
      }

      console.log(`[embed] Reused ${cachedIndices.length} embeddings from local cache`);
    }

    // For uncached: call the embedding provider
    if (uncachedIndices.length > 0) {
      const textsToEmbed = uncachedIndices.map((i) => state.combinedTexts[i]);

      console.log(
        `[embed] Generating ${textsToEmbed.length} new embeddings with ${embeddingProvider.name}/${embeddingProvider.model}`,
      );

      try {
        const newEmbeddings = await embeddingProvider.embedBatch(textsToEmbed);
        for (let j = 0; j < uncachedIndices.length; j++) {
          embeddings[uncachedIndices[j]] = newEmbeddings[j];
        }
        console.log(`[embed] Generated ${newEmbeddings.length} embeddings`);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error(`[embed] Batch embedding failed: ${msg}`);
        const errors = uncachedIndices.map((i) => ({
          sectionId: state.rows[i].section_id,
          error: `Embedding error: ${msg}`,
        }));
        // Keep any cached embeddings, only mark uncached as errors
        const validEmbeddings = embeddings.filter((e): e is number[] => e !== null);
        const validRows = state.rows.filter((_, i) => embeddings[i] !== null);
        const validParsed = state.parsedJsons.filter((_, i) => embeddings[i] !== null);
        const validTexts = state.combinedTexts.filter((_, i) => embeddings[i] !== null);
        return {
          rows: validRows,
          parsedJsons: validParsed,
          combinedTexts: validTexts,
          embeddings: validEmbeddings,
          errorCount: uncachedIndices.length,
          errors,
        };
      }
    }

    return { embeddings: embeddings.filter((e): e is number[] => e !== null) };
  }

  // --- Node 5: Cache embeddings to disk IMMEDIATELY ---
  async function cacheEmbeddings(state: PipelineState): Promise<Partial<PipelineState>> {
    if (state.done || state.embeddings.length === 0 || state.rows.length === 0) {
      return { cachedCount: 0 };
    }

    // Only cache rows that aren't already in the cache
    const newRows: SqliteRow[] = [];
    const newTexts: string[] = [];
    const newEmbeddings: number[][] = [];

    for (let i = 0; i < state.rows.length; i++) {
      if (!cache.has(state.rows[i].section_id)) {
        newRows.push(state.rows[i]);
        newTexts.push(state.combinedTexts[i]);
        newEmbeddings.push(state.embeddings[i]);
      }
    }

    if (newRows.length > 0) {
      cache.saveBatch(
        newRows,
        newTexts,
        newEmbeddings,
        embeddingProvider.name,
        embeddingProvider.model,
      );
      console.log(`[cache] Saved ${newRows.length} new embeddings to disk`);
    }

    return { cachedCount: newRows.length };
  }

  // --- Node 6: Save to PostgreSQL ---
  async function saveToPostgres(state: PipelineState): Promise<Partial<PipelineState>> {
    if (state.done || state.embeddings.length === 0) {
      return { batchOffset: state.batchOffset + state.batchSize };
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
    .addNode("filterAlreadyProcessed", filterAlreadyProcessed)
    .addNode("parseAndBuildText", parseAndBuildText)
    .addNode("generateEmbeddings", generateEmbeddings)
    .addNode("cacheEmbeddings", cacheEmbeddings)
    .addNode("saveToPostgres", saveToPostgres)
    // Edges
    .addEdge(START, "readFromSqlite")
    .addEdge("readFromSqlite", "filterAlreadyProcessed")
    .addEdge("filterAlreadyProcessed", "parseAndBuildText")
    .addEdge("parseAndBuildText", "generateEmbeddings")
    .addEdge("generateEmbeddings", "cacheEmbeddings")
    .addEdge("cacheEmbeddings", "saveToPostgres")
    .addConditionalEdges("saveToPostgres", shouldContinue);

  return graph.compile();
}
