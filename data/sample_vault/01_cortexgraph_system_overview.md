---
title: System Overview & Architecture
tags: [architecture, second-brain, langgraph, chromadb]
created: 2026-10-15T09:00:00Z
source_type: markdown
---

# CortexGraph Architecture

CortexGraph is an Agentic Second Brain and Personal Knowledge Assistant built on top of LangGraph, OpenAI models, and ChromaDB.

## Core Pillars

The system revolves around three core architectural pillars:

### Stateful Conversation Memory
Stateful conversation memory is maintained via LangGraph SQLite/Postgres checkpointers, guaranteeing determinism across restarts.

### Corrective RAG Pipeline
The Corrective RAG (CRAG) pipeline uses two-stage document parsing and ChromaDB vector retrieval to answer knowledge queries with grounded citations.

### End-to-End Telemetry
End-to-end telemetry and observability are provided by structlog JSON logging and native Langfuse tracing.

## Subsystem Interactions

### AST Chunking & Markdown Preservation
The system parses Obsidian and Logseq Markdown vaults into structured AST chunks preserving heading hierarchies. See [[adr-002-two-stage-chunking-chromadb]] for technical details on how Markdown headers are preserved.

### State Persistence & Checkpointing
Conversation turns are persisted deterministically as described in [[langgraph-checkpointer-persistence]].
