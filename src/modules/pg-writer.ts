/**
 * PostgreSQL writer module using Prisma.
 *
 * Upserts section data across 8 normalized tables:
 *   Section, SectionContext, SectionAesthetic, SectionComposition,
 *   SectionUiElements, SectionContentStrategy, SectionDesignTokens,
 *   SectionEmbedding
 *
 * Embedding vector is written via raw SQL since Prisma can't handle
 * the pgvector `vector` type natively.
 */

import { PrismaClient } from "../generated/prisma/client.js";
import type { InputJsonValue } from "../generated/prisma/internal/prismaNamespace.js";
import type {
  SqliteRow,
  RawJsonParsed,
  ParsedContext,
  ParsedAesthetic,
  ParsedComposition,
  ParsedUiElements,
  ParsedContentStrategy,
  ParsedDesignTokens,
} from "../types/index.js";
import { PrismaPg } from '@prisma/adapter-pg'

// Import the actual enum objects (not just types) so we can validate at runtime
import {
  $Enums,
} from "../generated/prisma/client.js";

// Convenience type aliases
type IndustryVertical = $Enums.IndustryVertical;
type PageType = $Enums.PageType;
type SectionType = $Enums.SectionType;
type Viewport = $Enums.Viewport;
type OverallVibe = $Enums.OverallVibe;
type ThemeMode = $Enums.ThemeMode;
type ColorPalette = $Enums.ColorPalette;
type BackgroundTreatment = $Enums.BackgroundTreatment;
type LayoutStructure = $Enums.LayoutStructure;
type SectionDivider = $Enums.SectionDivider;
type TypographyStyle = $Enums.TypographyStyle;
type TextDecoration = $Enums.TextDecoration;
type IconStyle = $Enums.IconStyle;
type IllustrationStyle = $Enums.IllustrationStyle;
type ProductImagery = $Enums.ProductImagery;
type NavigationStyle = $Enums.NavigationStyle;
type ContainerStyle = $Enums.ContainerStyle;
type ButtonStyle = $Enums.ButtonStyle;
type AnimationIndicator = $Enums.AnimationIndicator;
type ToneOfVoice = $Enums.ToneOfVoice;
type VisualToTextRatio = $Enums.VisualToTextRatio;
type ContentFormat = $Enums.ContentFormat;
type BorderRadius = $Enums.BorderRadius;
type SpacingDensity = $Enums.SpacingDensity;
type GlassmorphismIntensity = $Enums.GlassmorphismIntensity;
type ShadowElevation = $Enums.ShadowElevation;
type ContrastLevel = $Enums.ContrastLevel;

// ---------------------------------------------------------------------------
// Build valid-value Sets from the generated enum objects for runtime validation
// ---------------------------------------------------------------------------

function enumSet(enumObj: Record<string, string>): Set<string> {
  return new Set(Object.values(enumObj));
}

const VALID: Record<string, Set<string>> = {
  IndustryVertical: enumSet($Enums.IndustryVertical),
  PageType: enumSet($Enums.PageType),
  SectionType: enumSet($Enums.SectionType),
  Viewport: enumSet($Enums.Viewport),
  OverallVibe: enumSet($Enums.OverallVibe),
  ThemeMode: enumSet($Enums.ThemeMode),
  ColorPalette: enumSet($Enums.ColorPalette),
  BackgroundTreatment: enumSet($Enums.BackgroundTreatment),
  LayoutStructure: enumSet($Enums.LayoutStructure),
  SectionDivider: enumSet($Enums.SectionDivider),
  TypographyStyle: enumSet($Enums.TypographyStyle),
  TextDecoration: enumSet($Enums.TextDecoration),
  IconStyle: enumSet($Enums.IconStyle),
  IllustrationStyle: enumSet($Enums.IllustrationStyle),
  ProductImagery: enumSet($Enums.ProductImagery),
  NavigationStyle: enumSet($Enums.NavigationStyle),
  ContainerStyle: enumSet($Enums.ContainerStyle),
  ButtonStyle: enumSet($Enums.ButtonStyle),
  AnimationIndicator: enumSet($Enums.AnimationIndicator),
  ToneOfVoice: enumSet($Enums.ToneOfVoice),
  VisualToTextRatio: enumSet($Enums.VisualToTextRatio),
  ContentFormat: enumSet($Enums.ContentFormat),
  BorderRadius: enumSet($Enums.BorderRadius),
  SpacingDensity: enumSet($Enums.SpacingDensity),
  GlassmorphismIntensity: enumSet($Enums.GlassmorphismIntensity),
  ShadowElevation: enumSet($Enums.ShadowElevation),
  ContrastLevel: enumSet($Enums.ContrastLevel),
};

// Collect dropped values once per run to avoid log spam
const _droppedWarnings = new Set<string>();

// ---------------------------------------------------------------------------
// Helpers: safe enum casting with validation
// ---------------------------------------------------------------------------

// Map from JSON values to Prisma enum names where they differ
// (Prisma can't have identifiers starting with digits)
const ENUM_REMAP: Record<string, string> = {
  "3d_render": "three_d_render",
  "3d": "three_d",
  "10_percent": "ten_percent",
  "2d_render": "two_d_render",
};

/**
 * Safely cast a string to a validated Prisma enum value.
 * Remaps known aliases, then checks against the valid set.
 * Returns undefined (and logs once) if the value isn't valid.
 */
function remapEnum<T extends string>(
  value: string | null | undefined,
  enumName: string,
): T | undefined {
  if (!value || value.trim() === "") return undefined;
  const mapped = ENUM_REMAP[value] ?? value;
  if (!VALID[enumName]?.has(mapped)) {
    const key = `${enumName}:${value}`;
    if (!_droppedWarnings.has(key)) {
      _droppedWarnings.add(key);
      console.warn(`[enum] Dropping unknown ${enumName} value: "${value}"`);
    }
    return undefined;
  }
  return mapped as T;
}

/**
 * Cast an array of strings to validated Prisma enum values.
 * Unknown values are silently filtered out (with one-time warning).
 */
function remapEnumArray<T extends string>(
  values: string[] | null | undefined,
  enumName: string,
): T[] {
  if (!values || !Array.isArray(values)) return [];
  const result: T[] = [];
  for (const v of values) {
    if (!v || v.trim() === "") continue;
    const mapped = ENUM_REMAP[v] ?? v;
    if (VALID[enumName]?.has(mapped)) {
      result.push(mapped as T);
    } else {
      const key = `${enumName}:${v}`;
      if (!_droppedWarnings.has(key)) {
        _droppedWarnings.add(key);
        console.warn(`[enum] Dropping unknown ${enumName} value: "${v}"`);
      }
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// PgWriter
// ---------------------------------------------------------------------------

export class PgWriter {
  private prisma: PrismaClient;

  constructor(prisma?: PrismaClient) {
    const connectionString = `${process.env.DATABASE_URL}`

    const adapter = new PrismaPg({ connectionString })
    this.prisma = prisma ?? new PrismaClient({ adapter });
  }

  /** Enable the pgvector extension. Call once before any vector operations. */
  async ensureVectorExtension(): Promise<void> {
    await this.prisma.$executeRawUnsafe(
      `CREATE EXTENSION IF NOT EXISTS vector;`,
    );
  }

  /** Create HNSW index for fast cosine similarity search. Idempotent. */
  async createVectorIndex(): Promise<void> {
    await this.prisma.$executeRawUnsafe(
      `CREATE INDEX IF NOT EXISTS section_embeddings_vector_idx
       ON section_embeddings
       USING hnsw (embedding vector_cosine_ops);`,
    );
  }

  /**
   * Upsert a single section + all related tables + embedding.
   * Runs inside a transaction so it's all-or-nothing.
   */
  async upsertSection(
    row: SqliteRow,
    parsed: RawJsonParsed,
    combinedText: string,
    embedding: number[],
    providerName: string,
    modelName: string,
  ): Promise<void> {
    await this.prisma.$transaction(async (tx) => {
      // 1. Core section
      await tx.section.upsert({
        where: { sectionId: row.section_id },
        create: {
          sectionId: row.section_id,
          fileKey: row.file_key,
          rawJson: JSON.parse(row.raw_json) as InputJsonValue,
          description: parsed.description || row.description || null,
          classificationNotes: parsed.classification_notes || row.classification_notes || null,
          modelUsed: row.model_used,
          status: row.status,
          errorMessage: row.error_message,
          processedAt: new Date(row.processed_at),
          createdAt: new Date(row.created_at),
        },
        update: {
          fileKey: row.file_key,
          rawJson: JSON.parse(row.raw_json) as InputJsonValue,
          description: parsed.description || row.description || null,
          classificationNotes: parsed.classification_notes || row.classification_notes || null,
          modelUsed: row.model_used,
          status: row.status,
          errorMessage: row.error_message,
        },
      });

      // 2. Context
      await this.upsertContext(tx, row.section_id, parsed.context);

      // 3. Aesthetic
      await this.upsertAesthetic(tx, row.section_id, parsed.aesthetic);

      // 4. Composition
      await this.upsertComposition(tx, row.section_id, parsed.composition);

      // 5. UI Elements
      await this.upsertUiElements(tx, row.section_id, parsed.ui_elements);

      // 6. Content Strategy
      await this.upsertContentStrategy(tx, row.section_id, parsed.content_strategy);

      // 7. Design Tokens
      await this.upsertDesignTokens(tx, row.section_id, parsed.design_tokens_specs);

      // 8. Embedding record (without vector — Prisma can't write vector type)
      await tx.sectionEmbedding.upsert({
        where: { sectionId: row.section_id },
        create: {
          sectionId: row.section_id,
          combinedText,
          provider: providerName,
          model: modelName,
          embeddedAt: new Date(),
        },
        update: {
          combinedText,
          provider: providerName,
          model: modelName,
          embeddedAt: new Date(),
        },
      });
    });

    // 9. Write embedding vector via raw SQL (outside transaction — Prisma
    //    doesn't support Unsupported types in transactions either)
    const vectorStr = `[${embedding.join(",")}]`;
    await this.prisma.$executeRawUnsafe(
      `UPDATE section_embeddings SET embedding = $1::vector WHERE section_id = $2`,
      vectorStr,
      row.section_id,
    );
  }

  // -------------------------------------------------------------------------
  // Sub-table upserts
  // -------------------------------------------------------------------------

  private async upsertContext(
    tx: Parameters<Parameters<PrismaClient["$transaction"]>[0]>[0],
    sectionId: string,
    ctx: ParsedContext,
  ): Promise<void> {
    const data = {
      industryVerticals: remapEnumArray<IndustryVertical>(ctx.industry_vertical, "IndustryVertical"),
      pageType: remapEnum<PageType>(ctx.page_type, "PageType"),
      sectionType: remapEnum<SectionType>(ctx.section_type, "SectionType"),
      viewport: remapEnum<Viewport>(ctx.viewport, "Viewport"),
    };

    await (tx as any).sectionContext.upsert({
      where: { sectionId },
      create: { sectionId, ...data },
      update: data,
    });
  }

  private async upsertAesthetic(
    tx: Parameters<Parameters<PrismaClient["$transaction"]>[0]>[0],
    sectionId: string,
    aes: ParsedAesthetic,
  ): Promise<void> {
    const data = {
      overallVibes: remapEnumArray<OverallVibe>(aes.overall_vibe, "OverallVibe"),
      themeMode: remapEnum<ThemeMode>(aes.theme_mode, "ThemeMode"),
      colorPaletteStyles: remapEnumArray<ColorPalette>(aes.color_palette_style, "ColorPalette"),
      backgroundTreatments: remapEnumArray<BackgroundTreatment>(aes.background_treatment, "BackgroundTreatment"),
    };

    await (tx as any).sectionAesthetic.upsert({
      where: { sectionId },
      create: { sectionId, ...data },
      update: data,
    });
  }

  private async upsertComposition(
    tx: Parameters<Parameters<PrismaClient["$transaction"]>[0]>[0],
    sectionId: string,
    comp: ParsedComposition,
  ): Promise<void> {
    const data = {
      layoutStructures: remapEnumArray<LayoutStructure>(comp.layout_structure, "LayoutStructure"),
      sectionDividers: remapEnum<SectionDivider>(comp.section_dividers, "SectionDivider"),
    };

    await (tx as any).sectionComposition.upsert({
      where: { sectionId },
      create: { sectionId, ...data },
      update: data,
    });
  }

  private async upsertUiElements(
    tx: Parameters<Parameters<PrismaClient["$transaction"]>[0]>[0],
    sectionId: string,
    ui: ParsedUiElements,
  ): Promise<void> {
    const data = {
      typographyStyles: remapEnumArray<TypographyStyle>(ui.typography_style, "TypographyStyle"),
      textDecorations: remapEnumArray<TextDecoration>(ui.text_decoration_details, "TextDecoration"),
      iconStyles: remapEnumArray<IconStyle>(ui.icon_style, "IconStyle"),
      illustrationStyles: remapEnumArray<IllustrationStyle>(ui.illustration_style, "IllustrationStyle"),
      productImageryTypes: remapEnumArray<ProductImagery>(ui.product_imagery_type, "ProductImagery"),
      navigationStyles: remapEnumArray<NavigationStyle>(ui.navigation_style, "NavigationStyle"),
      containerStyles: remapEnumArray<ContainerStyle>(ui.container_style, "ContainerStyle"),
      buttonStyles: remapEnumArray<ButtonStyle>(ui.button_style, "ButtonStyle"),
      animationIndicators: remapEnumArray<AnimationIndicator>(ui.animation_indicators, "AnimationIndicator"),
    };

    await (tx as any).sectionUiElements.upsert({
      where: { sectionId },
      create: { sectionId, ...data },
      update: data,
    });
  }

  private async upsertContentStrategy(
    tx: Parameters<Parameters<PrismaClient["$transaction"]>[0]>[0],
    sectionId: string,
    cs: ParsedContentStrategy,
  ): Promise<void> {
    const data = {
      tonesOfVoice: remapEnumArray<ToneOfVoice>(cs.tone_of_voice, "ToneOfVoice"),
      visualToTextRatio: remapEnum<VisualToTextRatio>(cs.visual_to_text_ratio, "VisualToTextRatio"),
      contentFormats: remapEnumArray<ContentFormat>(cs.content_format, "ContentFormat"),
    };

    await (tx as any).sectionContentStrategy.upsert({
      where: { sectionId },
      create: { sectionId, ...data },
      update: data,
    });
  }

  private async upsertDesignTokens(
    tx: Parameters<Parameters<PrismaClient["$transaction"]>[0]>[0],
    sectionId: string,
    dt: ParsedDesignTokens,
  ): Promise<void> {
    const data = {
      borderRadius: remapEnum<BorderRadius>(dt.border_radius_scale, "BorderRadius"),
      spacingDensity: remapEnum<SpacingDensity>(dt.spacing_density, "SpacingDensity"),
      glassmorphismIntensity: remapEnum<GlassmorphismIntensity>(dt.glassmorphism_intensity, "GlassmorphismIntensity"),
      shadowElevation: remapEnum<ShadowElevation>(dt.shadow_elevation, "ShadowElevation"),
      contrastLevel: remapEnum<ContrastLevel>(dt.contrast_level, "ContrastLevel"),
    };

    await (tx as any).sectionDesignTokens.upsert({
      where: { sectionId },
      create: { sectionId, ...data },
      update: data,
    });
  }

  // -------------------------------------------------------------------------
  // Batch upsert
  // -------------------------------------------------------------------------

  async upsertBatch(
    rows: SqliteRow[],
    parsedJsons: RawJsonParsed[],
    combinedTexts: string[],
    embeddings: number[][],
    providerName: string,
    modelName: string,
  ): Promise<{ success: number; errors: Array<{ sectionId: string; error: string }> }> {
    let success = 0;
    const errors: Array<{ sectionId: string; error: string }> = [];

    for (let i = 0; i < rows.length; i++) {
      try {
        await this.upsertSection(
          rows[i],
          parsedJsons[i],
          combinedTexts[i],
          embeddings[i],
          providerName,
          modelName,
        );
        success++;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        errors.push({ sectionId: rows[i].section_id, error: msg });
      }
    }

    return { success, errors };
  }

  // -------------------------------------------------------------------------
  // Skip-detection: find sections already fully processed
  // -------------------------------------------------------------------------

  /**
   * Given a list of section IDs, return the subset that already exist in PG
   * with a completed embedding (i.e. section_embeddings row with non-null vector).
   *
   * This is the source of truth for "successfully saved" — if the embedding
   * vector is present, we know the entire transaction succeeded.
   */
  async getAlreadyProcessedIds(sectionIds: string[]): Promise<Set<string>> {
    if (sectionIds.length === 0) return new Set();

    const results = await this.prisma.$queryRawUnsafe<
      Array<{ section_id: string }>
    >(
      `SELECT section_id
       FROM section_embeddings
       WHERE section_id = ANY($1)
         AND embedding IS NOT NULL`,
      sectionIds,
    );

    return new Set(results.map((r) => r.section_id));
  }

  // -------------------------------------------------------------------------
  // Similarity search
  // -------------------------------------------------------------------------

  /**
   * Find the N most similar sections to a query embedding.
   * Returns section_id, combined_text, description, and cosine similarity.
   */
  async similaritySearch(
    queryEmbedding: number[],
    topK = 10,
  ): Promise<
    Array<{
      section_id: string;
      combined_text: string;
      description: string | null;
      similarity: number;
    }>
  > {
    const vectorStr = `[${queryEmbedding.join(",")}]`;
    return this.prisma.$queryRawUnsafe<
      Array<{
        section_id: string;
        combined_text: string;
        description: string | null;
        similarity: number;
      }>
    >(
      `SELECT
         se.section_id,
         se.combined_text,
         s.description,
         1 - (se.embedding <=> $1::vector) as similarity
       FROM section_embeddings se
       JOIN sections s ON s.section_id = se.section_id
       WHERE se.embedding IS NOT NULL
       ORDER BY se.embedding <=> $1::vector
       LIMIT $2`,
      vectorStr,
      topK,
    );
  }

  async disconnect(): Promise<void> {
    await this.prisma.$disconnect();
  }

  get client(): PrismaClient {
    return this.prisma;
  }
}
