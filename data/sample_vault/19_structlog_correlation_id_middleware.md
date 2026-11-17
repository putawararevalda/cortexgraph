---
title: "Structlog Correlation ID Middleware and Contextvars Binding"
tags: [structlog, logging, telemetry, contextvars]
created: 2026-10-31T15:00:00Z
source_type: markdown
---

# Structlog Correlation Middleware

## Correlation IDs

### Request & Trace Identifiers
Every inbound HTTP request extracts or generates:
- `X-Request-Id`: Unique per HTTP roundtrip.
- `X-Trace-Id`: Shared with Langfuse root trace ID.
- `thread_id`: Bound from conversation state.

### Single-Line JSON Formatting
Logs are structured as single-line JSON in production, preserving field types.

## Contextvars Binding

### Async-Safe Context Propagation
Using Python `contextvars`, correlation IDs flow automatically across async task boundaries and LangGraph node executions.
See [[langfuse-tracing-runnable-config]] and [[resiliency-tenacity-exponential-backoff]].
