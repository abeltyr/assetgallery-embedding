/**
 * Text builder module.
 *
 * Combines description + classification_notes + the 6 feature objects
 * into a single detailed text string suitable for embedding.
 *
 * Uses the strongly-typed parsed interfaces rather than generic Records.
 * For each feature object, properties are iterated:
 *   - strings are appended directly
 *   - arrays are joined with ", "
 */

import type {
  RawJsonParsed,
  ParsedContext,
  ParsedAesthetic,
  ParsedComposition,
  ParsedUiElements,
  ParsedContentStrategy,
  ParsedDesignTokens,
} from "../types/index.js";

// ---------------------------------------------------------------------------
// Feature-specific text builders
// ---------------------------------------------------------------------------

function contextToText(ctx: ParsedContext): string {
  const parts: string[] = [];
  if (ctx.industry_vertical?.length)
    parts.push(`industry vertical is ${ctx.industry_vertical.join(", ")}`);
  if (ctx.page_type) parts.push(`page type is ${ctx.page_type}`);
  if (ctx.section_type) parts.push(`section type is ${ctx.section_type}`);
  if (ctx.viewport) parts.push(`viewport is ${ctx.viewport}`);
  return `Context: ${parts.join(". ")}.`;
}

function aestheticToText(aes: ParsedAesthetic): string {
  const parts: string[] = [];
  if (aes.overall_vibe?.length)
    parts.push(`overall vibe is ${aes.overall_vibe.join(", ")}`);
  if (aes.theme_mode) parts.push(`theme mode is ${aes.theme_mode}`);
  if (aes.color_palette_style?.length)
    parts.push(`color palette style is ${aes.color_palette_style.join(", ")}`);
  if (aes.background_treatment?.length)
    parts.push(`background treatment is ${aes.background_treatment.join(", ")}`);
  return `Aesthetic: ${parts.join(". ")}.`;
}

function compositionToText(comp: ParsedComposition): string {
  const parts: string[] = [];
  if (comp.layout_structure?.length)
    parts.push(`layout structure is ${comp.layout_structure.join(", ")}`);
  if (comp.section_dividers)
    parts.push(`section dividers is ${comp.section_dividers}`);
  return `Composition: ${parts.join(". ")}.`;
}

function uiElementsToText(ui: ParsedUiElements): string {
  const parts: string[] = [];
  if (ui.typography_style?.length)
    parts.push(`typography style is ${ui.typography_style.join(", ")}`);
  if (ui.text_decoration_details?.length)
    parts.push(`text decoration is ${ui.text_decoration_details.join(", ")}`);
  if (ui.icon_style?.length)
    parts.push(`icon style is ${ui.icon_style.join(", ")}`);
  if (ui.illustration_style?.length)
    parts.push(`illustration style is ${ui.illustration_style.join(", ")}`);
  if (ui.product_imagery_type?.length)
    parts.push(`product imagery type is ${ui.product_imagery_type.join(", ")}`);
  if (ui.navigation_style?.length)
    parts.push(`navigation style is ${ui.navigation_style.join(", ")}`);
  if (ui.container_style?.length)
    parts.push(`container style is ${ui.container_style.join(", ")}`);
  if (ui.button_style?.length)
    parts.push(`button style is ${ui.button_style.join(", ")}`);
  if (ui.animation_indicators?.length)
    parts.push(`animation indicators is ${ui.animation_indicators.join(", ")}`);
  return `UI Elements: ${parts.join(". ")}.`;
}

function contentStrategyToText(cs: ParsedContentStrategy): string {
  const parts: string[] = [];
  if (cs.tone_of_voice?.length)
    parts.push(`tone of voice is ${cs.tone_of_voice.join(", ")}`);
  if (cs.visual_to_text_ratio)
    parts.push(`visual to text ratio is ${cs.visual_to_text_ratio}`);
  if (cs.content_format?.length)
    parts.push(`content format is ${cs.content_format.join(", ")}`);
  return `Content Strategy: ${parts.join(". ")}.`;
}

function designTokensToText(dt: ParsedDesignTokens): string {
  const parts: string[] = [];
  if (dt.border_radius_scale)
    parts.push(`border radius scale is ${dt.border_radius_scale}`);
  if (dt.spacing_density)
    parts.push(`spacing density is ${dt.spacing_density}`);
  if (dt.glassmorphism_intensity)
    parts.push(`glassmorphism intensity is ${dt.glassmorphism_intensity}`);
  if (dt.shadow_elevation)
    parts.push(`shadow elevation is ${dt.shadow_elevation}`);
  if (dt.contrast_level)
    parts.push(`contrast level is ${dt.contrast_level}`);
  return `Design Tokens: ${parts.join(". ")}.`;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Build the combined text from a parsed raw_json object.
 *
 * Order:
 *   1. Description
 *   2. Classification notes (if present)
 *   3. Context
 *   4. Aesthetic
 *   5. Composition
 *   6. UI Elements
 *   7. Content Strategy
 *   8. Design Tokens
 */
export function buildCombinedText(parsed: RawJsonParsed): string {
  const sections: string[] = [];

  if (parsed.description) {
    sections.push(parsed.description);
  }

  if (parsed.classification_notes) {
    sections.push(`Classification Notes: ${parsed.classification_notes}`);
  }

  sections.push(contextToText(parsed.context));
  sections.push(aestheticToText(parsed.aesthetic));
  sections.push(compositionToText(parsed.composition));
  sections.push(uiElementsToText(parsed.ui_elements));
  sections.push(contentStrategyToText(parsed.content_strategy));
  sections.push(designTokensToText(parsed.design_tokens_specs));

  return sections.join("\n\n");
}

/**
 * Parse the raw_json string from SQLite into a strongly-typed object.
 */
export function parseRawJson(rawJsonStr: string): RawJsonParsed {
  const p = JSON.parse(rawJsonStr);

  return {
    context: {
      industry_vertical: Array.isArray(p.context?.industry_vertical)
        ? p.context.industry_vertical
        : [],
      page_type: p.context?.page_type ?? "",
      section_type: p.context?.section_type ?? "",
      viewport: p.context?.viewport ?? undefined,
    },
    aesthetic: {
      overall_vibe: Array.isArray(p.aesthetic?.overall_vibe)
        ? p.aesthetic.overall_vibe
        : [],
      theme_mode: p.aesthetic?.theme_mode ?? "",
      color_palette_style: Array.isArray(p.aesthetic?.color_palette_style)
        ? p.aesthetic.color_palette_style
        : [],
      background_treatment: Array.isArray(p.aesthetic?.background_treatment)
        ? p.aesthetic.background_treatment
        : [],
    },
    composition: {
      layout_structure: Array.isArray(p.composition?.layout_structure)
        ? p.composition.layout_structure
        : [],
      section_dividers: p.composition?.section_dividers ?? "",
    },
    ui_elements: {
      typography_style: Array.isArray(p.ui_elements?.typography_style)
        ? p.ui_elements.typography_style
        : [],
      text_decoration_details: Array.isArray(p.ui_elements?.text_decoration_details)
        ? p.ui_elements.text_decoration_details
        : [],
      icon_style: Array.isArray(p.ui_elements?.icon_style)
        ? p.ui_elements.icon_style
        : [],
      illustration_style: Array.isArray(p.ui_elements?.illustration_style)
        ? p.ui_elements.illustration_style
        : [],
      product_imagery_type: Array.isArray(p.ui_elements?.product_imagery_type)
        ? p.ui_elements.product_imagery_type
        : [],
      navigation_style: Array.isArray(p.ui_elements?.navigation_style)
        ? p.ui_elements.navigation_style
        : [],
      container_style: Array.isArray(p.ui_elements?.container_style)
        ? p.ui_elements.container_style
        : [],
      button_style: Array.isArray(p.ui_elements?.button_style)
        ? p.ui_elements.button_style
        : [],
      animation_indicators: Array.isArray(p.ui_elements?.animation_indicators)
        ? p.ui_elements.animation_indicators
        : [],
    },
    content_strategy: {
      tone_of_voice: Array.isArray(p.content_strategy?.tone_of_voice)
        ? p.content_strategy.tone_of_voice
        : [],
      visual_to_text_ratio: p.content_strategy?.visual_to_text_ratio ?? "",
      content_format: Array.isArray(p.content_strategy?.content_format)
        ? p.content_strategy.content_format
        : [],
    },
    design_tokens_specs: {
      border_radius_scale: p.design_tokens_specs?.border_radius_scale ?? "",
      spacing_density: p.design_tokens_specs?.spacing_density ?? "",
      glassmorphism_intensity: p.design_tokens_specs?.glassmorphism_intensity ?? "",
      shadow_elevation: p.design_tokens_specs?.shadow_elevation ?? "",
      contrast_level: p.design_tokens_specs?.contrast_level ?? "",
    },
    description: p.description ?? "",
    classification_notes: p.classification_notes ?? "",
  };
}
