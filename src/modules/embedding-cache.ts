/**
 * Local embedding cache — guarantees no embedding is ever lost.
 *
 * Stores embeddings as JSONL (one JSON object per line) in a local file.
 * Each entry contains everything needed to retry a PG save without
 * calling the embedding API again.
 *
 * File format (one per line):
 *   {"section_id":"...","combined_text":"...","embedding":[...],"provider":"openai","model":"text-embedding-3-large","raw_json":"...","file_key":"...","cached_at":"..."}
 *
 * On startup, the cache loads all section_ids into memory for O(1) lookups.
 * New entries are appended synchronously to guarantee durability even if
 * the process crashes immediately after.
 */

import fs from "fs";
import path from "path";
import type { SqliteRow, RawJsonParsed } from "../types/index.js";

export interface CachedEmbedding {
  section_id: string;
  combined_text: string;
  embedding: number[];
  provider: string;
  model: string;
  raw_json: string;
  file_key: string;
  model_used: string;
  description: string | null;
  classification_notes: string | null;
  processed_at: string;
  created_at: string;
  status: string;
  error_message: string | null;
  cached_at: string;
}

const CACHE_DIR = ".cache";
const CACHE_FILE = "embeddings.jsonl";

export class EmbeddingCache {
  private filePath: string;
  private knownIds: Set<string>;

  constructor(cacheDir?: string) {
    const dir = cacheDir ?? path.resolve(CACHE_DIR);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    this.filePath = path.join(dir, CACHE_FILE);
    this.knownIds = new Set();
    this.loadIndex();
  }

  /** Load all section_ids from the cache file into memory */
  private loadIndex(): void {
    if (!fs.existsSync(this.filePath)) return;

    const content = fs.readFileSync(this.filePath, "utf-8");
    for (const line of content.split("\n")) {
      if (!line.trim()) continue;
      try {
        const obj = JSON.parse(line);
        if (obj.section_id) {
          this.knownIds.add(obj.section_id);
        }
      } catch {
        // skip corrupt lines
      }
    }

    console.log(`[cache] Loaded ${this.knownIds.size} cached embeddings from ${this.filePath}`);
  }

  /** Check if a section_id already has a cached embedding */
  has(sectionId: string): boolean {
    return this.knownIds.has(sectionId);
  }

  /** Return the set of all cached section IDs */
  getCachedIds(): Set<string> {
    return new Set(this.knownIds);
  }

  /**
   * Persist a batch of embeddings to the cache file.
   * This is synchronous and uses appendFileSync to guarantee the data
   * hits disk before we return — no embedding can be lost.
   */
  saveBatch(
    rows: SqliteRow[],
    combinedTexts: string[],
    embeddings: number[][],
    provider: string,
    model: string,
  ): void {
    const lines: string[] = [];
    const now = new Date().toISOString();

    for (let i = 0; i < rows.length; i++) {
      const entry: CachedEmbedding = {
        section_id: rows[i].section_id,
        combined_text: combinedTexts[i],
        embedding: embeddings[i],
        provider,
        model,
        raw_json: rows[i].raw_json,
        file_key: rows[i].file_key,
        model_used: rows[i].model_used,
        description: rows[i].description,
        classification_notes: rows[i].classification_notes,
        processed_at: rows[i].processed_at,
        created_at: rows[i].created_at,
        status: rows[i].status,
        error_message: rows[i].error_message,
        cached_at: now,
      };
      lines.push(JSON.stringify(entry));
      this.knownIds.add(rows[i].section_id);
    }

    // Synchronous write — guaranteed on disk before we continue
    fs.appendFileSync(this.filePath, lines.join("\n") + "\n", "utf-8");
  }

  /**
   * Read all cached entries that haven't been saved to PG yet.
   * Takes a set of already-processed IDs (from PG) and returns
   * only the entries that still need to be saved.
   */
  getUnsavedEntries(alreadySavedIds: Set<string>): CachedEmbedding[] {
    if (!fs.existsSync(this.filePath)) return [];

    const entries: CachedEmbedding[] = [];
    const content = fs.readFileSync(this.filePath, "utf-8");
    // Dedupe: if the same section_id appears multiple times
    // (e.g. from multiple runs), keep only the latest
    const seen = new Map<string, CachedEmbedding>();

    for (const line of content.split("\n")) {
      if (!line.trim()) continue;
      try {
        const entry = JSON.parse(line) as CachedEmbedding;
        if (entry.section_id && !alreadySavedIds.has(entry.section_id)) {
          seen.set(entry.section_id, entry);
        }
      } catch {
        // skip corrupt lines
      }
    }

    for (const entry of seen.values()) {
      entries.push(entry);
    }

    return entries;
  }

  get path(): string {
    return this.filePath;
  }

  get size(): number {
    return this.knownIds.size;
  }
}
