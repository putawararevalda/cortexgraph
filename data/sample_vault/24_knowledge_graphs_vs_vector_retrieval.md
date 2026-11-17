---
title: "Knowledge Graphs vs Vector Store Retrieval in Personal Notes"
tags: [knowledge-graphs, vectorstore, hybrid-search, rag]
created: 2026-11-05T12:00:00Z
source_type: markdown
---

# Knowledge Graphs vs Vector Stores

## Comparison

### Vector Store Strengths & Weaknesses
- **Vector Stores (ChromaDB):** Excel at fuzzy semantic lookup, conceptual similarity, and unstructured natural language queries. Weak at deterministic multi-hop relationship traversals.

### Knowledge Graph Strengths & Weaknesses
- **Knowledge Graphs:** Excel at explicit relationship queries (e.g., "Find all ADRs authored by Revalda that mention LangGraph").

## Hybrid Approach in CortexGraph

### Bidirectional WikiLink Extraction
We extract bidirectional `[[WikiLinks]]` during AST parsing and store them in chunk metadata.

### Relational Context Augmentation
Allows traversing connected notes alongside vector similarity search.
See [[obsidian-vault-linking-and-wikilinks]] and [[adr-002-two-stage-chunking-chromadb]].
