---
title: "LangGraph Checkpointer Persistence: SQLite and Postgres"
tags: [langgraph, persistence, checkpointer, sqlite]
created: 2026-10-19T11:00:00Z
source_type: markdown
---

# LangGraph Checkpointer Persistence

## Architecture

### Decoupled State Storage
LangGraph state persistence decouples conversation threads from web server worker processes. Each thread is addressed by a unique `configurable={"thread_id": "..."}` key.

### Transactional Atomicity
State snapshots are committed after every node execution, ensuring crash resilience and replayability.

## Implementation Details

### AsyncSqliteSaver for Local Development
We use `AsyncSqliteSaver.from_conn_string("./data/cortex_state.db")` for zero-dependency local testing.

### AsyncPostgresSaver for Production
We support `AsyncPostgresSaver` backed by connection pooling for production deployments.
See [[adr-001-langgraph-state-machine]] and [[fastapi-sse-streaming-architecture]].
