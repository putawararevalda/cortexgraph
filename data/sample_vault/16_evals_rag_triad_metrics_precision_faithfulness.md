---
title: "RAG Triad Evaluation: Precision, Faithfulness, and Relevance"
tags: [evals, metrics, faithfulness, benchmarks]
created: 2026-10-28T09:20:00Z
source_type: markdown
---

# RAG Triad Evaluation Metrics

## Core Dimensions

### Context Precision (Target >= 0.85)
Proportion of retrieved chunks that contain ground truth facts necessary to answer the question.

### Faithfulness (Target >= 0.90)
Proportion of claims in the generated response that are directly entailed by the retrieved context. Zero hallucinations allowed.

### Answer Relevance (Target >= 0.85)
Degree to which the generated answer directly addresses the original query without irrelevant digressions.
See [[corrective-rag-crag-architecture]] and [[reading-notes-self-rag-reflection]].
