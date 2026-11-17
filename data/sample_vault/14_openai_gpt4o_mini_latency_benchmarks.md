---
title: "OpenAI GPT-4o-mini Latency and Streaming Benchmarks"
tags: [openai, gpt4o-mini, latency, benchmarks]
created: 2026-10-26T16:00:00Z
source_type: markdown
---

# GPT-4o-mini Streaming Benchmarks

## Benchmark Results

### Time to First Token (TTFT)
- Time to First Token (TTFT): ~28-35 ms under warm connection.

### Token Generation Speed
- Token throughput: ~110 tokens/second.
- Structured output latency: ~120 ms for small JSON payloads.

## Impact on UX

### Perceived System Responsiveness
Fast TTFT enables perceived instantaneous response over FastAPI Server-Sent Events (SSE).

### Cost Reduction in Multi-Node CRAG
Cost reduction allows running 3 LLM calls per turn (router, grader, synthesizer) under $0.001 total cost.
See [[fastapi-sse-streaming-architecture]] and [[corrective-rag-crag-architecture]].
