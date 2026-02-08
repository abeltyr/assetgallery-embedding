/**
 * Shared types for the embedding pipeline.
 *
 * These mirror the exact output schema from data.txt and the Prisma enums.
 */

// =============================================================================
// SQLite source row
// =============================================================================

/** Raw row from the SQLite analysis_results table */
export interface SqliteRow {
  id: number;
  section_id: string;
  app_code: string;
  platform_code: string;
  version_code: string;
  page_code: string;
  section_code: string;
  file_key: string;
  raw_json: string;
  industry_vertical: string | null;
  page_type: string | null;
  section_type: string | null;
  theme_mode: string | null;
  description: string | null;
  classification_notes: string | null;
  model_used: string;
  status: string;
  error_message: string | null;
  processed_at: string;
  created_at: string;
}

// =============================================================================
// Parsed raw_json — strongly typed per data.txt output schema
// =============================================================================

export interface ParsedContext {
  industry_vertical: string[];
  page_type: string;
  section_type: string;
  viewport?: string;
}

export interface ParsedAesthetic {
  overall_vibe: string[];
  theme_mode: string;
  color_palette_style: string[];
  background_treatment: string[];
}

export interface ParsedComposition {
  layout_structure: string[];
  section_dividers: string;
}

export interface ParsedUiElements {
  typography_style: string[];
  text_decoration_details: string[];
  icon_style: string[];
  illustration_style: string[];
  product_imagery_type: string[];
  navigation_style: string[];
  container_style: string[];
  button_style: string[];
  animation_indicators: string[];
}

export interface ParsedContentStrategy {
  tone_of_voice: string[];
  visual_to_text_ratio: string;
  content_format: string[];
}

export interface ParsedDesignTokens {
  border_radius_scale: string;
  spacing_density: string;
  glassmorphism_intensity: string;
  shadow_elevation: string;
  contrast_level: string;
}

/** The 8 top-level keys in raw_json, fully typed */
export interface RawJsonParsed {
  context: ParsedContext;
  aesthetic: ParsedAesthetic;
  composition: ParsedComposition;
  ui_elements: ParsedUiElements;
  content_strategy: ParsedContentStrategy;
  design_tokens_specs: ParsedDesignTokens;
  description: string;
  classification_notes: string;
}

// =============================================================================
// Pipeline state
// =============================================================================

/** State flowing through the LangGraph pipeline */
export interface PipelineState {
  rows: SqliteRow[];
  parsedJsons: RawJsonParsed[];
  combinedTexts: string[];
  embeddings: number[][];
  processedCount: number;
  errorCount: number;
  errors: Array<{ sectionId: string; error: string }>;
}

// =============================================================================
// Embedding provider interface
// =============================================================================

/** Implement this to add new embedding providers */
export interface EmbeddingProvider {
  readonly name: string;
  readonly model: string;
  readonly dimensions: number;
  embed(text: string): Promise<number[]>;
  embedBatch(texts: string[]): Promise<number[][]>;
}
