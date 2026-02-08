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

// Re-export the generated enums so callers can use them for filtering
import type {
  IndustryVertical,
  PageType,
  SectionType,
  Viewport,
  OverallVibe,
  ThemeMode,
  ColorPalette,
  BackgroundTreatment,
  LayoutStructure,
  SectionDivider,
  TypographyStyle,
  TextDecoration,
  IconStyle,
  IllustrationStyle,
  ProductImagery,
  NavigationStyle,
  ContainerStyle,
  ButtonStyle,
  AnimationIndicator,
  ToneOfVoice,
  VisualToTextRatio,
  ContentFormat,
  BorderRadius,
  SpacingDensity,
  GlassmorphismIntensity,
  ShadowElevation,
  ContrastLevel,
} from "../generated/prisma/client.js";

// ---------------------------------------------------------------------------
// Helpers: safe enum casting
// ---------------------------------------------------------------------------

/**
 * Safely cast a string to a Prisma enum value.
 * Returns undefined if the value is null/empty/not a valid member.
 * We rely on Prisma to reject truly invalid values at write time.
 */
function toEnum<T extends string>(value: string | null | undefined): T | undefined {
  if (!value || value.trim() === "") return undefined;
  // Normalize: the JSON may use "3d_render" but the Prisma enum uses "three_d_render"
  return value as T;
}

/**
 * Cast an array of strings to an array of Prisma enum values.
 * Filters out empty/null entries.
 */
function toEnumArray<T extends string>(values: string[] | null | undefined): T[] {
  if (!values || !Array.isArray(values)) return [];
  return values.filter((v) => v && v.trim() !== "") as T[];
}

// Map from JSON values to Prisma enum names where they differ
// (Prisma can't have identifiers starting with digits)
const ENUM_REMAP: Record<string, string> = {
  "3d_render": "three_d_render",
  "3d": "three_d",
  "10_percent": "ten_percent",
};

function remapEnum<T extends string>(value: string | null | undefined): T | undefined {
  if (!value || value.trim() === "") return undefined;
  const mapped = ENUM_REMAP[value] ?? value;
  return mapped as T;
}

function remapEnumArray<T extends string>(values: string[] | null | undefined): T[] {
  if (!values || !Array.isArray(values)) return [];
  return values
    .filter((v) => v && v.trim() !== "")
    .map((v) => (ENUM_REMAP[v] ?? v) as T);
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
      industryVerticals: remapEnumArray<IndustryVertical>(ctx.industry_vertical),
      pageType: remapEnum<PageType>(ctx.page_type),
      sectionType: remapEnum<SectionType>(ctx.section_type),
      viewport: remapEnum<Viewport>(ctx.viewport),
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
      overallVibes: remapEnumArray<OverallVibe>(aes.overall_vibe),
      themeMode: remapEnum<ThemeMode>(aes.theme_mode),
      colorPaletteStyles: remapEnumArray<ColorPalette>(aes.color_palette_style),
      backgroundTreatments: remapEnumArray<BackgroundTreatment>(aes.background_treatment),
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
      layoutStructures: remapEnumArray<LayoutStructure>(comp.layout_structure),
      sectionDividers: remapEnum<SectionDivider>(comp.section_dividers),
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
      typographyStyles: remapEnumArray<TypographyStyle>(ui.typography_style),
      textDecorations: remapEnumArray<TextDecoration>(ui.text_decoration_details),
      iconStyles: remapEnumArray<IconStyle>(ui.icon_style),
      illustrationStyles: remapEnumArray<IllustrationStyle>(ui.illustration_style),
      productImageryTypes: remapEnumArray<ProductImagery>(ui.product_imagery_type),
      navigationStyles: remapEnumArray<NavigationStyle>(ui.navigation_style),
      containerStyles: remapEnumArray<ContainerStyle>(ui.container_style),
      buttonStyles: remapEnumArray<ButtonStyle>(ui.button_style),
      animationIndicators: remapEnumArray<AnimationIndicator>(ui.animation_indicators),
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
      tonesOfVoice: remapEnumArray<ToneOfVoice>(cs.tone_of_voice),
      visualToTextRatio: remapEnum<VisualToTextRatio>(cs.visual_to_text_ratio),
      contentFormats: remapEnumArray<ContentFormat>(cs.content_format),
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
      borderRadius: remapEnum<BorderRadius>(dt.border_radius_scale),
      spacingDensity: remapEnum<SpacingDensity>(dt.spacing_density),
      glassmorphismIntensity: remapEnum<GlassmorphismIntensity>(dt.glassmorphism_intensity),
      shadowElevation: remapEnum<ShadowElevation>(dt.shadow_elevation),
      contrastLevel: remapEnum<ContrastLevel>(dt.contrast_level),
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
