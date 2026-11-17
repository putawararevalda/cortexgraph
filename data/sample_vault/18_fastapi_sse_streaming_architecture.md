---
title: "FastAPI Server-Sent Events (SSE) Streaming Architecture"
tags: [fastapi, sse, streaming, architecture]
created: 2026-10-30T13:40:00Z
source_type: markdown
---

# FastAPI SSE Streaming Architecture

## Event Protocol

### Node Status Event
`event: node_status`: Emits `{"node": "chroma_retriever", "status": "running"}` to indicate graph progress.

### Token Streaming Event
`event: token`: Emits text token chunks as LLM generates content in real time.

### Citation & Final Payload
`event: citation`: Emits structured citation objects when synthesis finishes.
See [[langfuse-tracing-runnable-config]] and [[openai-gpt4o-mini-latency-benchmarks]].
