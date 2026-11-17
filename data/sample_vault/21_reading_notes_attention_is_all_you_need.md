---
title: "Reading Notes: Attention Is All You Need (Vaswani et al.)"
tags: [reading-notes, transformers, attention, deep-learning]
created: 2026-11-02T14:30:00Z
source_type: markdown
---

# Reading Notes: Attention Is All You Need

## Summary & Core Contributions

### Elimination of Recurrence
Introduced the Transformer architecture, replacing recurrent and convolutional layers with Multi-Head Self-Attention.

### Mathematical Formulation
Key mechanisms:
- Scaled Dot-Product Attention: $Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$.
- Positional encodings to inject sequence order without recurrence.

## Implications for Second Brain RAG

### Global Context Modeling
Transformers model all-to-all token relationships, enabling high-quality embeddings.

### Quadratic Memory Scaling
Quadratic memory scaling $O(N^2)$ dictates chunking long personal notes into bounded token windows (512 tokens).
See [[reading-notes-lost-in-the-middle-rag]] and [[adr-002-two-stage-chunking-chromadb]].
