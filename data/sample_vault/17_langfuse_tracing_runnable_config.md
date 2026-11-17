---
title: "Langfuse Tracing Instrumentation via RunnableConfig"
tags: [langfuse, observability, tracing, langgraph]
created: 2026-10-29T11:30:00Z
source_type: markdown
---

# Langfuse Tracing with LangGraph

## Integration Strategy

### RunnableConfig Injection
We inject the official `langfuse.langchain.CallbackHandler` into LangGraph invocation config:
```python
config = {
    "callbacks": [langfuse_handler],
    "configurable": {"thread_id": thread_id}
}
```

### Trace Hierarchies
Every LangGraph node automatically generates a child span under the root HTTP trace.

## Span Attribution

### Session & User Grouping
- Session grouping by `thread_id`.
- Scores and user feedback attached via `trace_id`.
See [[structlog-correlation-id-middleware]] and [[fastapi-sse-streaming-architecture]].
