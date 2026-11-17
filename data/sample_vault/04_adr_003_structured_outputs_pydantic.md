---
title: "ADR-003: OpenAI Structured Outputs with Pydantic"
tags: [adr, openai, pydantic, structured-outputs]
created: 2026-10-15T10:30:00Z
source_type: markdown
---

# ADR-003: OpenAI Structured Outputs

## Context

### Brittle JSON Parsing
Freeform text outputs from LLM grading and routing nodes require regex parsing and frequently fail on malformed JSON outputs.

### Schema Enforcement Needs
Routing and grading nodes require guaranteed schema conformity without retry overhead.

## Decision

### Native Pydantic Schemas
We enforce OpenAI Structured Outputs (`client.beta.chat.completions.parse` / `model.with_structured_output`) backed by strict Pydantic models:
- `IntentClassification`
- `QueryAnalysis`
- `DocumentGrade`
- `HallucinationGrade`

### Strict Schema Validation
We configure models with `strict=True` to eliminate schema validation errors during graph execution.

## Consequences

### Reliable State Machine Routing
100% reliable schema adherence for grading and routing. Zero regex extraction heuristics needed in production nodes.
See [[adr-001-langgraph-state-machine]] and [[corrective-rag-crag-architecture]].
