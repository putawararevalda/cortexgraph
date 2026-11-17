---
title: "Postmortem: Unbounded Context Overflow in Multi-Turn Chat"
tags: [postmortem, memory, context-window, debugging]
created: 2026-10-16T14:00:00Z
source_type: markdown
---

# Postmortem: Context Window Overflow

## Incident Summary

### Initial Trigger
On October 16, 2026, long multi-turn conversations exceeded the context window limit, resulting in API 400 Bad Request errors.

### User Impact
Active user sessions failed abruptly during extended knowledge exploration queries.

## Root Cause

### Unbounded Message Accumulation
The LangGraph state appended all messages via `add_messages` without trimming older turns or summarizing conversation history.

### Memory Leakage
State checkpoints grew to dozens of kilobytes, exceeding token budgets upon reload.

## Remediation

### Rolling Summarization Node
Implemented `memory_manager` node triggering summarization when conversation length exceeds 12 messages or 4,000 tokens.

### Message History Pruning
Pruned older turns using LangGraph `RemoveMessage` operations while preserving the latest 4 messages verbatim.
See [[langgraph-memory-manager-summarization]] and [[langgraph-checkpointer-persistence]].
