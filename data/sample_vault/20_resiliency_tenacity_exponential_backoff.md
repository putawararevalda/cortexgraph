---
title: "Resiliency and Exponential Backoff Retries with Tenacity"
tags: [resiliency, tenacity, retries, rate-limits]
created: 2026-11-01T10:15:00Z
source_type: markdown
---

# Tenacity Resiliency Patterns

## Rate Limits & Transient Faults

### OpenAI API Transient Errors
OpenAI API calls can fail due to HTTP 429 (rate limit exceeded) or transient 500/503 network timeouts.

### Connection Resilience
Client timeouts must be handled gracefully without crashing user conversational flows.

## Retry Policy

### Exponential Backoff & Jitter
- Stop after 3 attempts.
- Exponential backoff: `wait_exponential(multiplier=1, min=2, max=10)`.
- Random jitter added to avoid thundering herd problems.
See [[structlog-correlation-id-middleware]] and [[openai-gpt4o-mini-latency-benchmarks]].
