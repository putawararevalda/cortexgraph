---
title: "Reading Notes: Lost in the Middle - How Language Models Use Long Contexts"
tags: [reading-notes, context-window, rag, retrieval]
created: 2026-11-03T16:00:00Z
source_type: markdown
---

# Reading Notes: Lost in the Middle

## Paper Summary (Liu et al., 2023)

### Positional Bias in Context Windows
Language models are significantly better at accessing information located at the beginning or end of long input contexts.

### Performance Degradation on Middle Facts
Performance degrades sharply for facts positioned in the middle of long multi-document prompts.

## Architectural Takeaways for CortexGraph

### Re-Ranking & Chunk Ordering
1. Rank most relevant chunks at the top and bottom of synthesized prompt context.

### Bounded Retrieval Set
2. Keep retrieval top-k small (k=4 to 5) rather than dumping large chunk lists into the prompt.
See [[langgraph-memory-manager-summarization]] and [[corrective-rag-crag-architecture]].
