---
title: "OpenAI Embeddings: Dimension and Cost Tradeoffs"
tags: [openai, embeddings, vector, cost-optimization]
created: 2026-10-25T14:20:00Z
source_type: markdown
---

# OpenAI Embeddings: Tradeoffs

## Models Compared

### text-embedding-3-small
- Dimensions: 1536 (supports truncation to 512).
- Cost: $0.02 per 1M tokens.
- MTEB benchmark: 62.3% retrieval accuracy.

### text-embedding-3-large
- Dimensions: 3072.
- Cost: $0.13 per 1M tokens (6.5x more expensive).
- MTEB benchmark: 64.6% retrieval accuracy.

## Conclusion for CortexGraph

### Selected Embedding Model
We standardise on `text-embedding-3-small` with 1536 dimensions.

### Cost-Efficiency Rationale
Provides optimal latency, low cost, and sufficient precision for personal note vaults under 50,000 notes.
See [[chromadb-cosine-similarity-indexing]] and [[openai-gpt4o-mini-latency-benchmarks]].
