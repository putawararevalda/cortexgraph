---
title: "SHA-256 Incremental Vault Synchronization"
tags: [chromadb, sync, sha256, ingestion]
created: 2026-10-23T09:15:00Z
source_type: markdown
---

# SHA-256 Incremental Vault Sync

## Motivation

### Avoiding Redundant Embeddings
Re-embedding an entire 10,000-note Obsidian vault on every minor edit wastes API credits and degrades ingestion performance.

### Sync Performance Targets
Full vault scans must complete in sub-second time when files remain unchanged.

## Deduplication Strategy (FR-03)

### Content Hash Calculation
1. Calculate SHA-256 `doc_hash` of file content before parsing.

### Hash Comparison & Skip
2. Query ChromaDB for existing chunks matching `source_path`.
3. If `existing_doc_hash == current_doc_hash`, skip parsing and embedding (0 API calls).

### Orphaned Chunk Purging
4. If changed or deleted, purge old chunks and upsert new AST chunks.
See [[adr-002-two-stage-chunking-chromadb]] and [[chromadb-cosine-similarity-indexing]].
