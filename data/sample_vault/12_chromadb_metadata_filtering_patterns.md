---
title: "ChromaDB Metadata Filtering and Relaxation Fallback"
tags: [chromadb, metadata, filtering, crag]
created: 2026-10-24T11:45:00Z
source_type: markdown
---

# ChromaDB Metadata Filtering

## Supported Metadata Filters

### Temporal Timestamp Bounding
Filter by Unix timestamp bounds: `created_after_ts`, `created_before_ts`.

### Categorical & Tag Matching
Filter by source type (`markdown`, `pdf`, `text`) and tags using `$contains` substring match on `,tag1,tag2,` formats.

## Relaxation Fallback

### Zero Result Recovery
If a strict metadata filter returns 0 chunks, CRAG automatically falls back to unfiltered semantic similarity search.

### Guarded Degradation
This prevents over-constrained queries from returning empty answers when relevant notes exist under related tags.
See [[chromadb-cosine-similarity-indexing]] and [[corrective-rag-crag-architecture]].
