/**
 * Main entrypoint for the embedding pipeline.
 *
 * Usage:
 *   npx tsx src/index.ts                   # Run the full pipeline
 *   npx tsx src/index.ts --retry           # Retry failed PG saves from local cache (no re-embedding)
 *   npx tsx src/index.ts --search "query"  # Search for similar sections
 *   npx tsx src/index.ts --dry-run         # Parse + build text only (no embed/save)
 */

import "dotenv/config";
import path from "path";
import { SqliteReader } from "./modules/sqlite-reader.js";
import { PgWriter } from "./modules/pg-writer.js";
import { createEmbeddingProvider } from "./modules/embedding-provider.js";
import { createEmbeddingPipeline } from "./graph/pipeline.js";
import { EmbeddingCache } from "./modules/embedding-cache.js";
import { parseRawJson, buildCombinedText } from "./modules/text-builder.js";

async function main() {
  const args = process.argv.slice(2);
  const isDryRun = args.includes("--dry-run");
  const isRetry = args.includes("--retry");
  const searchIdx = args.indexOf("--search");
  const searchQuery = searchIdx >= 0 ? args[searchIdx + 1] : null;

  const sqlitePath = path.resolve(
    process.env.SQLITE_DB_PATH ?? "./analysis_results.db",
  );
  const batchSize = parseInt(process.env.BATCH_SIZE ?? "50", 10);

  console.log("=== Embedding Pipeline ===");
  console.log(`SQLite DB:  ${sqlitePath}`);
  console.log(`Provider:   ${process.env.EMBEDDING_PROVIDER ?? "openai"}`);
  console.log(`Batch size: ${batchSize}`);
  console.log();

  // --- Search mode ---
  if (searchQuery) {
    console.log(`Searching for: "${searchQuery}"`);
    const embeddingProvider = createEmbeddingProvider();
    const pgWriter = new PgWriter();

    try {
      const queryEmbedding = await embeddingProvider.embed(searchQuery);
      const results = await pgWriter.similaritySearch(queryEmbedding, 10);

      console.log(`\nFound ${results.length} results:\n`);
      for (const r of results) {
        console.log(`  [${r.similarity.toFixed(4)}] ${r.section_id}`);
        console.log(`    ${r.description?.substring(0, 120) ?? "(no description)"}...`);
        console.log();
      }
    } finally {
      await pgWriter.disconnect();
    }
    return;
  }

  // --- Dry run mode ---
  if (isDryRun) {
    console.log("DRY RUN — will parse and build text but NOT embed or save\n");
    const reader = new SqliteReader(sqlitePath);

    try {
      const rows = reader.readBatch(0, 5);
      console.log(`Sample of ${rows.length} rows:\n`);

      for (const row of rows) {
        const parsed = parseRawJson(row.raw_json);
        const text = buildCombinedText(parsed);
        console.log(`--- ${row.section_id} ---`);
        console.log(text.substring(0, 500));
        console.log("...\n");
      }
    } finally {
      reader.close();
    }
    return;
  }

  // --- Retry mode: replay cached embeddings into PG without re-embedding ---
  if (isRetry) {
    console.log("RETRY MODE — replaying cached embeddings into PostgreSQL\n");

    const cache = new EmbeddingCache();
    const pgWriter = new PgWriter();

    try {
      await pgWriter.ensureVectorExtension();

      // Find what's in PG already
      const allPgIds = await pgWriter.getAlreadyProcessedIds(
        [...cache.getCachedIds()],
      );

      const unsaved = cache.getUnsavedEntries(allPgIds);
      console.log(`Found ${unsaved.length} cached embeddings not yet in PG\n`);

      if (unsaved.length === 0) {
        console.log("Nothing to retry. All cached embeddings are already in PG.");
        return;
      }

      let success = 0;
      let errors = 0;
      const errorDetails: Array<{ sectionId: string; error: string }> = [];
      const startTime = Date.now();

      for (const entry of unsaved) {
        try {
          // Reconstruct the SqliteRow-like object from cache
          const row = {
            id: 0,
            section_id: entry.section_id,
            app_code: "",
            platform_code: "",
            version_code: "",
            page_code: "",
            section_code: "",
            file_key: entry.file_key,
            raw_json: entry.raw_json,
            industry_vertical: null,
            page_type: null,
            section_type: null,
            theme_mode: null,
            description: entry.description,
            classification_notes: entry.classification_notes,
            model_used: entry.model_used,
            status: entry.status,
            error_message: entry.error_message,
            processed_at: entry.processed_at,
            created_at: entry.created_at,
          };

          const parsed = parseRawJson(entry.raw_json);

          await pgWriter.upsertSection(
            row,
            parsed,
            entry.combined_text,
            entry.embedding,
            entry.provider,
            entry.model,
          );
          success++;
        } catch (err) {
          errors++;
          const msg = err instanceof Error ? err.message : String(err);
          errorDetails.push({ sectionId: entry.section_id, error: msg });
        }
      }

      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      await pgWriter.createVectorIndex();

      console.log("\n=== Retry Complete ===");
      console.log(`Saved:   ${success}`);
      console.log(`Errors:  ${errors}`);
      console.log(`Time:    ${elapsed}s`);

      if (errorDetails.length > 0) {
        console.log("\nErrors:");
        for (const e of errorDetails.slice(0, 10)) {
          console.log(`  ${e.sectionId}: ${e.error}`);
        }
        if (errorDetails.length > 10) {
          console.log(`  ... and ${errorDetails.length - 10} more`);
        }
      }
    } finally {
      await pgWriter.disconnect();
    }
    return;
  }

  // --- Full pipeline mode ---
  const reader = new SqliteReader(sqlitePath);
  const pgWriter = new PgWriter();
  const embeddingProvider = createEmbeddingProvider();
  const cache = new EmbeddingCache();

  try {
    // Ensure pgvector extension and table exist
    await pgWriter.ensureVectorExtension();

    const totalValid = reader.getValidCount();
    const totalAll = reader.getTotalCount();
    console.log(`Total rows: ${totalAll}, valid for processing: ${totalValid}`);
    console.log(`Local cache: ${cache.size} embeddings on disk\n`);

    if (totalValid === 0) {
      console.log("No valid rows to process. Exiting.");
      return;
    }

    // Create and run the LangGraph pipeline
    const pipeline = createEmbeddingPipeline(reader, pgWriter, embeddingProvider, cache);

    const startTime = Date.now();

    const finalState = await pipeline.invoke({
      batchOffset: 0,
      batchSize,
      totalValid,
    }, { recursionLimit: 10000 });

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

    // Create vector index after data is loaded
    await pgWriter.createVectorIndex();

    console.log("\n=== Pipeline Complete ===");
    console.log(`Processed: ${finalState.processedCount}`);
    console.log(`Skipped:   ${finalState.skippedCount} (already in PG)`);
    console.log(`Cached:    ${finalState.cachedCount} (new embeddings saved to disk)`);
    console.log(`Errors:    ${finalState.errorCount}`);
    console.log(`Time:      ${elapsed}s`);
    console.log(`Cache:     ${cache.path} (${cache.size} total)`);

    if (finalState.errorCount > 0) {
      console.log(`\nRun with --retry to replay failed saves from cache.`);
    }

    if (finalState.errors.length > 0) {
      console.log("\nErrors:");
      for (const e of finalState.errors.slice(0, 10)) {
        console.log(`  ${e.sectionId}: ${e.error}`);
      }
      if (finalState.errors.length > 10) {
        console.log(`  ... and ${finalState.errors.length - 10} more`);
      }
    }
  } finally {
    reader.close();
    await pgWriter.disconnect();
  }
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
