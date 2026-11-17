---
title: "ADR-002: Two-Stage Document Chunking for ChromaDB Vector Store"
tags: [adr, rag, chunking, chromadb]
created: 2026-10-15T10:00:00Z
source_type: markdown
---

# ADR-002: Two-Stage Document Chunking

## Context

### Heading Loss in Naive Splitting
Naive character splitting breaks Markdown header hierarchies and loses crucial context when indexing technical notes and architecture decision records.

### Retrieval Degradation
Without section context, chunks retrieved by similarity search lack semantic anchors, causing inaccurate synthesis.

## Decision

### Stage 1: Structural AST Splitting
We implement Stage 1 structural splitting using `MarkdownHeaderTextSplitter` on `#`, `##`, and `###` headers, establishing a breadcrumb heading hierarchy.

### Stage 2: Recursive Character Splitting
We implement Stage 2 semantic bounding using `RecursiveCharacterTextSplitter.from_tiktoken_encoder` with a 512 token target and 64 token overlap.

## Consequences

### Breadcrumb Context Retention
Chunks retain their breadcrumb heading paths (`[File: ... | Section: ...]`). Citations can accurately reference specific section anchors in the personal vault.
See [[chromadb-cosine-similarity-indexing]] and [[chromadb-incremental-sha256-sync]].
