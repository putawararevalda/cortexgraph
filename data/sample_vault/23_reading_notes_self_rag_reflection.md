---
title: "Reading Notes: Self-RAG - Learning to Retrieve, Generate, and Critique"
tags: [reading-notes, self-rag, reflection, evaluation]
created: 2026-11-04T10:45:00Z
source_type: markdown
---

# Reading Notes: Self-RAG (Asai et al., 2023)

## Overview

### Self-Reflection Tokens
Self-RAG trains models to output reflection tokens:
- `[Retrieve]`: Decide when external retrieval is needed.
- `[IsRel]`: Evaluate whether retrieved chunks are relevant.
- `[IsSup]`: Verify if generation is supported by retrieved evidence.

### Dynamic Generation Control
Allows adaptive retrieval only when the model lacks factual certainty.

## Adoption in CortexGraph

### Modular LangGraph Node Topology
Rather than fine-tuning bespoke models, we implement Self-RAG principles via explicit LangGraph nodes:
- `intent_router` implements `[Retrieve]` logic.
- `document_grader` implements `[IsRel]` logic.
- `hallucination_grader` implements `[IsSup]` logic.
See [[corrective-rag-crag-architecture]] and [[evals-rag-triad-metrics-precision-faithfulness]].
