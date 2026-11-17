---
title: "Cyclic Graphs and Recursion Limits in LangGraph"
tags: [langgraph, cyclic, circuit-breaker, algorithms]
created: 2026-10-20T13:15:00Z
source_type: markdown
---

# Cyclic Graphs and Recursion Limits

## The Need for Cycles

### Corrective Feedback Loops
Corrective RAG inherently requires cycles: when retrieved documents are insufficient, the query must be rewritten and re-executed against the vector store.

### Self-Correction Mechanism
When the hallucination grader flags ungrounded statements, the answer must be regenerated from verified evidence.

## Circuit Breakers

### Rewrite Attempt Bounding
`rewrite_count` limits query rewriting to 2 iterations before forcing synthesis or direct fallback.

### Recursion Depth Limits
LangGraph `recursion_limit` is set to 25 to protect against edge misconfigurations.
See [[corrective-rag-crag-architecture]] and [[postmortem-2026-11-rag-hallucination-loop]].
