---
title: "Postmortem: Cyclic Hallucination Loop in RAG Generation"
tags: [postmortem, rag, hallucination, circuit-breaker]
created: 2026-11-16T16:30:00Z
source_type: markdown
---

# Postmortem: Cyclic Hallucination Loop

## Incident Summary

### Infinite Regeneration Loop
During test runs on out-of-domain queries, the hallucination grader repeatedly failed the synthesizer output, triggering an infinite loop of regeneration.

### Recursion Limit Trigger
The graph hit LangGraph recursion limits (25 steps), aborting execution with unhandled GraphRecursionError.

## Root Cause

### Absence of Counter Guards
The conditional edge `route_after_hallucination_check` routed back to `rag_synthesizer` without checking an explicit iteration counter.

### Missing Out-of-Domain Guardrail
When the knowledge vault contained 0 relevant chunks, the synthesizer attempted to fabricate an answer from base model knowledge.

## Remediation

### Circuit Breakers on Regeneration
Added `generation_count` to `AgentState` and enforced a strict circuit breaker of `MAX_RETRIES = 2`.

### Explicit Abstention Guardrails
Added an explicit abstention guardrail: if 0 relevant chunks are retrieved, the synthesizer states lack of information rather than attempting generation.
See [[langgraph-cyclic-graphs-and-recursion-limits]] and [[corrective-rag-crag-architecture]].
