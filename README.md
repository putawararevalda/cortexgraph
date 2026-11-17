# CortexGraph

> **Agentic Second Brain & Personal Knowledge Assistant**  
> Built with LangGraph, OpenAI, ChromaDB, FastAPI, and Langfuse.

## Overview

CortexGraph is an autonomous knowledge assistant that indexes your markdown notes, performs corrective RAG (CRAG) retrieval, grades context relevance, self-corrects hallucinations, and maintains structured conversational memory.

## Architecture

```text
[User Request]
       │
       ▼
 [intent_router] ──(Direct Chitchat)──► [direct_responder] ──► [memory_manager]
       │
   (RAG Need)
       ▼
[query_analyzer] ──► [chroma_retriever] ──► [document_grader]
                                                    │
                                  ┌─────────────────┴─────────────────┐
                            (Sufficient)                         (Insufficient)
                                  │                                    │
                                  ▼                                    ▼
                          [rag_synthesizer] ◄───(Rewrite)─── [query_rewriter]
                                  │                              (Max 2 Loops)
                                  ▼
                        [hallucination_grader]
                                  │
                 ┌────────────────┴────────────────┐
             (Grounded)                       (Ungrounded)
                 │                                 │
                 ▼                                 ▼
         [memory_manager]                   [rag_synthesizer]
                 │                           (Regeneration)
                 ▼
          [Final Response]
```

### Compiled LangGraph Mermaid Topology

```mermaid
graph TD;
	__start__([start]):::first
	intent_router(intent_router)
	direct_responder(direct_responder)
	query_analyzer(query_analyzer)
	chroma_retriever(chroma_retriever)
	document_grader(document_grader)
	query_rewriter(query_rewriter)
	rag_synthesizer(rag_synthesizer)
	hallucination_grader(hallucination_grader)
	memory_manager(memory_manager)
	__end__([end]):::last
	__start__ --> intent_router;
	chroma_retriever --> document_grader;
	direct_responder --> memory_manager;
	document_grader -.-> query_rewriter;
	document_grader -.-> rag_synthesizer;
	hallucination_grader -.-> memory_manager;
	hallucination_grader -.-> rag_synthesizer;
	intent_router -.-> direct_responder;
	intent_router -.-> query_analyzer;
	query_analyzer --> chroma_retriever;
	query_rewriter --> chroma_retriever;
	rag_synthesizer --> hallucination_grader;
	memory_manager --> __end__;
```

## Quick Start

### 1. Prerequisites
- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (recommended) or standard `pip`
- Docker & Docker Compose (optional for local ChromaDB)

### 2. Environment Setup
```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Install Dependencies
```bash
make setup
```

### 4. Run Tests & Linter
```bash
make test
make lint
```

### 5. Run Application
```bash
make run
```
Access the Swagger UI at `http://localhost:8000/docs` and healthcheck at `http://localhost:8000/health`.

### 6. Containerized Deployment
```bash
docker compose up --build
```
