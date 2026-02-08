/**
 * SQLite reader module.
 *
 * Opens the source SQLite database and reads rows from the
 * `analysis_results` table in configurable batches.
 */

import Database from "better-sqlite3";
import type { SqliteRow } from "../types/index.js";

export class SqliteReader {
  private db: Database.Database;
  private tableName: string;

  constructor(dbPath: string, tableName = "analysis_results") {
    this.db = new Database(dbPath, { readonly: true });
    this.tableName = tableName;
  }

  /** Get total row count */
  getTotalCount(): number {
    const row = this.db
      .prepare(`SELECT COUNT(*) as count FROM ${this.tableName}`)
      .get() as { count: number };
    return row.count;
  }

  /**
   * Read rows in batches using LIMIT/OFFSET.
   * Only returns rows with status = 'success' and non-null raw_json.
   */
  readBatch(offset: number, limit: number): SqliteRow[] {
    return this.db
      .prepare(
        `SELECT * FROM ${this.tableName}
         WHERE status = 'success' AND raw_json IS NOT NULL
         ORDER BY id ASC
         LIMIT ? OFFSET ?`,
      )
      .all(limit, offset) as SqliteRow[];
  }

  /** Read all valid rows */
  readAll(): SqliteRow[] {
    return this.db
      .prepare(
        `SELECT * FROM ${this.tableName}
         WHERE status = 'success' AND raw_json IS NOT NULL
         ORDER BY id ASC`,
      )
      .all() as SqliteRow[];
  }

  /** Get count of valid rows (success + non-null raw_json) */
  getValidCount(): number {
    const row = this.db
      .prepare(
        `SELECT COUNT(*) as count FROM ${this.tableName}
         WHERE status = 'success' AND raw_json IS NOT NULL`,
      )
      .get() as { count: number };
    return row.count;
  }

  close(): void {
    this.db.close();
  }
}
