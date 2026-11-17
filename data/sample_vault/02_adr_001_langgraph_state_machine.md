---
title: "ADR-001: Explicit LangGraph State Machine vs Autonomous Loop"
tags: [adr, architecture, langgraph, decision]
created: 2026-10-15T09:30:00Z
source_type: markdown
---

# ADR-001: Explicit LangGraph State Machine

## Context

### Autonomous Agent Limitations
Autonomous agent frameworks like AutoGPT often suffer from unbounded execution loops, state drift, and difficult debugging in production environments.

### Production Constraints
Personal knowledge assistants require predictable execution bounds, deterministic routing, and zero hallucination risk.

## Decision

### Strongly Typed State Transitions
We chose an explicit LangGraph StateGraph topology with strongly typed state transitions. Every node updates a partial dictionary adhering to `AgentState`.

### Circuit Breakers on Conditional Edges
Circuit breakers such as `MAX_RETRIES = 2` are enforced on conditional edges to prevent infinite query rewriting or regeneration loops.

## Consequences

### Determinism & Traceability
Deterministic execution guarantees that the agent terminates predictably. Full state history can be checkpointed and reloaded across server restarts. Traceability in Langfuse is simplified because every node maps to a distinct trace span.
See [[adr-002-two-stage-chunking-chromadb]] and [[langgraph-cyclic-graphs-and-recursion-limits]].
