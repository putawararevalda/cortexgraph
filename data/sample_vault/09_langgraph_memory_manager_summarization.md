---
title: "Conversation Memory Management and Rolling Summarization"
tags: [memory, summarization, langgraph, llm]
created: 2026-10-21T15:45:00Z
source_type: markdown
---

# Conversation Memory Management

## Problem Definition

### Linear Token Escalation
Chat histories grow indefinitely. As context expands, token costs increase linearly.

### Attention Degradation
Attention degrades across long contexts ("Lost in the Middle"), leading to forgotten system instructions.

## The Rolling Summarization Algorithm

### Threshold Evaluation
1. Evaluates total message count (> 12) or token count (> 4,000 tokens).

### Condensation & Pruning
2. Condenses messages prior to the latest 4 into a running summary.
3. Emits `RemoveMessage` operations to prune the state graph message list.
See [[reading-notes-lost-in-the-middle-rag]] and [[postmortem-2026-10-context-overflow]].
