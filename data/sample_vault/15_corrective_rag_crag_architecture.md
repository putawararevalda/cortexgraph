---
title: "Corrective RAG (CRAG) Pipeline Topology"
tags: [crag, rag, architecture, langgraph]
created: 2026-10-27T10:10:00Z
source_type: markdown
---

# Corrective RAG (CRAG) Pipeline

## Execution Flow

### Step 1: Retrieval & Relevance Grading
1. `chroma_retriever`: Fetches top-k chunks with metadata filters.
2. `document_grader`: Evaluates relevance of each chunk via structured output (`is_relevant: bool`).

### Step 2: Query Rewriting or Synthesis
3. Conditional branch:
   - If any chunk is relevant: Proceed to `rag_synthesizer`.
   - If 0 chunks relevant: Route to `query_rewriter` (loops max 2 times).

### Step 3: Synthesis & Hallucination Check
4. `rag_synthesizer`: Generates answer with mandatory inline citations.
5. `hallucination_grader`: Verifies factual grounding before outputting to user.
See [[adr-001-langgraph-state-machine]] and [[reading-notes-self-rag-reflection]].
