/**
 * Embedding provider module — swappable between OpenAI, Cohere, and Ollama.
 *
 * Each provider implements the EmbeddingProvider interface so the pipeline
 * doesn't care which one is active.
 */

import type { EmbeddingProvider } from "../types/index.js";

// ---------------------------------------------------------------------------
// OpenAI provider
// ---------------------------------------------------------------------------
export class OpenAIEmbeddingProvider implements EmbeddingProvider {
  readonly name = "openai";
  readonly model: string;
  readonly dimensions: number;
  private apiKey: string;

  constructor(
    apiKey: string,
    model = "text-embedding-3-small",
    dimensions = 1536,
  ) {
    this.apiKey = apiKey;
    this.model = model;
    this.dimensions = dimensions;
  }

  async embed(text: string): Promise<number[]> {
    const [result] = await this.embedBatch([text]);
    return result;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    const response = await fetch("https://api.openai.com/v1/embeddings", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        input: texts,
        model: this.model,
        dimensions: this.dimensions,
      }),
    });

    if (!response.ok) {
      const err = await response.text();
      throw new Error(`OpenAI embedding API error (${response.status}): ${err}`);
    }

    const json = (await response.json()) as {
      data: Array<{ embedding: number[]; index: number }>;
    };

    // Sort by index to preserve input order
    return json.data
      .sort((a, b) => a.index - b.index)
      .map((d) => d.embedding);
  }
}

// ---------------------------------------------------------------------------
// Cohere provider
// ---------------------------------------------------------------------------
export class CohereEmbeddingProvider implements EmbeddingProvider {
  readonly name = "cohere";
  readonly model: string;
  readonly dimensions: number;
  private apiKey: string;

  constructor(
    apiKey: string,
    model = "embed-english-v3.0",
    dimensions = 1024,
  ) {
    this.apiKey = apiKey;
    this.model = model;
    this.dimensions = dimensions;
  }

  async embed(text: string): Promise<number[]> {
    const [result] = await this.embedBatch([text]);
    return result;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    const response = await fetch("https://api.cohere.ai/v1/embed", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        texts,
        model: this.model,
        input_type: "search_document",
        truncate: "END",
      }),
    });

    if (!response.ok) {
      const err = await response.text();
      throw new Error(`Cohere embedding API error (${response.status}): ${err}`);
    }

    const json = (await response.json()) as {
      embeddings: number[][];
    };

    return json.embeddings;
  }
}

// ---------------------------------------------------------------------------
// Ollama (local) provider
// ---------------------------------------------------------------------------
export class OllamaEmbeddingProvider implements EmbeddingProvider {
  readonly name = "ollama";
  readonly model: string;
  readonly dimensions: number;
  private baseUrl: string;

  constructor(
    baseUrl = "http://localhost:11434",
    model = "nomic-embed-text",
    dimensions = 768,
  ) {
    this.baseUrl = baseUrl;
    this.model = model;
    this.dimensions = dimensions;
  }

  async embed(text: string): Promise<number[]> {
    const response = await fetch(`${this.baseUrl}/api/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: this.model, prompt: text }),
    });

    if (!response.ok) {
      const err = await response.text();
      throw new Error(`Ollama embedding API error (${response.status}): ${err}`);
    }

    const json = (await response.json()) as { embedding: number[] };
    return json.embedding;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    // Ollama doesn't support native batching, so we parallelize single calls
    return Promise.all(texts.map((t) => this.embed(t)));
  }
}

// ---------------------------------------------------------------------------
// Factory — create the right provider from env config
// ---------------------------------------------------------------------------
export function createEmbeddingProvider(
  provider?: string,
): EmbeddingProvider {
  const name = (provider ?? process.env.EMBEDDING_PROVIDER ?? "openai").toLowerCase();

  switch (name) {
    case "openai": {
      const apiKey = process.env.OPENAI_API_KEY;
      if (!apiKey || apiKey === "sk-...") {
        throw new Error("OPENAI_API_KEY is not set in .env");
      }
      return new OpenAIEmbeddingProvider(
        apiKey,
        process.env.OPENAI_EMBEDDING_MODEL ?? "text-embedding-3-small",
        1536,
      );
    }

    case "cohere": {
      const apiKey = process.env.COHERE_API_KEY;
      if (!apiKey) {
        throw new Error("COHERE_API_KEY is not set in .env");
      }
      return new CohereEmbeddingProvider(
        apiKey,
        process.env.COHERE_EMBEDDING_MODEL ?? "embed-english-v3.0",
        1024,
      );
    }

    case "ollama": {
      return new OllamaEmbeddingProvider(
        process.env.OLLAMA_BASE_URL ?? "http://localhost:11434",
        process.env.OLLAMA_EMBEDDING_MODEL ?? "nomic-embed-text",
        768,
      );
    }

    default:
      throw new Error(
        `Unknown embedding provider "${name}". Supported: openai, cohere, ollama`,
      );
  }
}
