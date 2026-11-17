---
title: "ChromaDB Cosine Similarity Indexing"
tags: [chromadb, vectorstore, cosine, hnsw]
created: 2026-10-22T08:30:00Z
source_type: markdown
---

# ChromaDB Cosine Similarity Indexing

## Collection Setup

### Space Configuration
The `second_brain_notes` collection is initialized with cosine distance:
```python
collection = client.get_or_create_collection(
    name="second_brain_notes",
    metadata={"hnsw:space": "cosine"}
)
```

### HNSW Parameter Tuning
HNSW index parameters are configured for fast sub-millisecond retrieval across personal knowledge bases.

## Distance vs Similarity

### Cosine Distance Inversion
ChromaDB returns cosine distance $d \in [0, 2]$.

### Normalized Score Mapping
We convert this to similarity score $s = 1.0 - d$, bounded in $[0.0, 1.0]$, ensuring intuitive relevance thresholds.
See [[chromadb-metadata-filtering-patterns]] and [[openai-embeddings-dimension-tradeoffs]].
