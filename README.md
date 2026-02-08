# Embedding Pipeline

LangGraph-based pipeline that reads UI section classification data from a SQLite database, generates embeddings, and stores everything in PostgreSQL with pgvector for similarity search.

## Architecture

```
SQLite (source)
  │
  ▼
┌─────────────────────────────────────────────────┐
│  LangGraph Pipeline                             │
│                                                 │
│  readFromSqlite → parseAndBuildText →           │
│  generateEmbeddings → saveToPostgres  ──loop──► │
│                                                 │
└─────────────────────────────────────────────────┘
  │
  ▼
PostgreSQL + pgvector (8 normalized tables)
```

## Database Schema

The data is **normalized across 8 tables** instead of a single flat table. Every array field in the JSON becomes a native PostgreSQL enum array column, and every single-value field becomes an enum column — all directly filterable without JSON parsing.

```
sections                    (core: sectionId, fileKey, rawJson, description, notes, metadata)
  ├── section_contexts      (industryVerticals[], pageType, sectionType, viewport)
  ├── section_aesthetics    (overallVibes[], themeMode, colorPaletteStyles[], backgroundTreatments[])
  ├── section_compositions  (layoutStructures[], sectionDividers)
  ├── section_ui_elements   (typographyStyles[], textDecorations[], iconStyles[], illustrationStyles[],
  │                          productImageryTypes[], navigationStyles[], containerStyles[],
  │                          buttonStyles[], animationIndicators[])
  ├── section_content_strategies (tonesOfVoice[], visualToTextRatio, contentFormats[])
  ├── section_design_tokens (borderRadius, spacingDensity, glassmorphismIntensity, shadowElevation, contrastLevel)
  └── section_embeddings    (combinedText, embedding vector(1536), provider, model)
```

All sub-tables link to `sections` via `section_id` (1:1, CASCADE delete).

### Why this design

- **Filterable**: `WHERE 'crypto' = ANY(industry_verticals)` — no JSON parsing
- **Searchable**: every enum column can be indexed, filtered, aggregated
- **Separated concerns**: embedding lives in its own table — you can re-embed without touching classification data
- **No duplicates**: `appCode`, `platformCode`, etc. are excluded (they live elsewhere in your system)
- **Type-safe**: 30+ Prisma enums match the strict enum list from `data.txt`

### Enum coverage

All enums from `data.txt` are defined in the Prisma schema: `IndustryVertical`, `PageType`, `SectionType`, `Viewport`, `OverallVibe`, `ThemeMode`, `ColorPalette`, `BackgroundTreatment`, `LayoutStructure`, `SectionDivider`, `TypographyStyle`, `TextDecoration`, `IconStyle`, `IllustrationStyle`, `ProductImagery`, `NavigationStyle`, `ContainerStyle`, `ButtonStyle`, `AnimationIndicator`, `ToneOfVoice`, `VisualToTextRatio`, `ContentFormat`, `BorderRadius`, `SpacingDensity`, `GlassmorphismIntensity`, `ShadowElevation`, `ContrastLevel`.

## Combined text construction

The embedding is generated from a combined text built in this order:

1. **Description** — the free-text visual description
2. **Classification notes** — optional ambiguity notes
3. **Context** — industry vertical, page type, section type
4. **Aesthetic** — vibe, theme, color palette, background
5. **Composition** — layout, dividers
6. **UI Elements** — all 9 sub-fields (typography, icons, buttons, etc.)
7. **Content Strategy** — tone, ratio, format
8. **Design Tokens** — radius, spacing, glassmorphism, shadow, contrast

For each feature object, string values are added directly and array values are joined with `, `.

## Prerequisites

- Node.js 18+
- PostgreSQL with [pgvector](https://github.com/pgvector/pgvector) extension
- SQLite database with `analysis_results` table
- An embedding API key (OpenAI by default)

## Setup

```bash
npm install

cp .env.example .env
# Edit .env: DATABASE_URL, OPENAI_API_KEY, SQLITE_DB_PATH

npm run db:generate
npm run db:push
```

## Usage

```bash
# Full pipeline
npm start

# Dry run (parse + print combined text, no embedding/saving)
npx tsx src/index.ts --dry-run

# Similarity search
npx tsx src/index.ts --search "minimalist dark mode hero section"
```

## Switching Embedding Providers

Set `EMBEDDING_PROVIDER` in `.env`:

| Provider | Value    | Env vars needed                            | Dimensions |
|----------|----------|--------------------------------------------|------------|
| OpenAI   | `openai` | `OPENAI_API_KEY`, `OPENAI_EMBEDDING_MODEL` | 1536       |
| Cohere   | `cohere` | `COHERE_API_KEY`, `COHERE_EMBEDDING_MODEL`  | 1024       |
| Ollama   | `ollama` | `OLLAMA_BASE_URL`, `OLLAMA_EMBEDDING_MODEL` | 768        |

If switching away from OpenAI, update the vector dimension in `prisma/schema.prisma`:

```prisma
embedding Unsupported("vector(1024)")? @map("embedding")  // Cohere
embedding Unsupported("vector(768)")?  @map("embedding")  // Ollama
```

Then `npm run db:push`.

## Project Structure

```
src/
├── index.ts                      # CLI entrypoint (3 modes)
├── types/index.ts                # Strongly-typed interfaces for all 6 JSON sub-objects
├── modules/
│   ├── embedding-provider.ts     # Swappable: OpenAI, Cohere, Ollama
│   ├── text-builder.ts           # Builds combined text from typed fields
│   ├── sqlite-reader.ts          # Reads batches from SQLite
│   └── pg-writer.ts              # Upserts across 8 PG tables + raw SQL for pgvector
└── graph/
    └── pipeline.ts               # LangGraph StateGraph (4 nodes + batch loop)
prisma/
└── schema.prisma                 # 8 models, 27 enums, pgvector extension
```

## Example Queries

```sql
-- Find all hero sections with dark theme and minimalist vibe
SELECT s.section_id, s.description
FROM sections s
JOIN section_contexts sc ON sc.section_id = s.section_id
JOIN section_aesthetics sa ON sa.section_id = s.section_id
WHERE sc.section_type = 'hero'
  AND sa.theme_mode = 'dark'
  AND 'minimalist' = ANY(sa.overall_vibes);

-- Find sections using glassmorphism with pill buttons
SELECT s.section_id
FROM sections s
JOIN section_design_tokens dt ON dt.section_id = s.section_id
JOIN section_ui_elements ui ON ui.section_id = s.section_id
WHERE dt.glassmorphism_intensity NOT IN ('none')
  AND 'pill' = ANY(ui.button_styles);
```
