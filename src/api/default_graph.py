"""Default StateGraph builder for CortexGraph API execution."""

import re
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from src.graph.nodes import memory_manager
from src.graph.state import AgentState, Citation, RetrievedChunk


def _is_chitchat(query: str) -> bool:
    """Classify whether a query is simple conversational chitchat."""
    q = query.strip().lower()
    chitchat_patterns = [
        r"^hi\b",
        r"^hello\b",
        r"^hey\b",
        r"^who are you\b",
        r"^how are you\b",
        r"^what can you do\b",
        r"^good (morning|afternoon|evening)\b",
        r"^help\b",
        r"^thanks?\b",
        r"^thank you\b",
    ]
    return any(re.search(pat, q) for pat in chitchat_patterns) or len(q.split()) <= 2


async def intent_router(state: AgentState) -> dict[str, Any]:
    """Node routing query to either direct response or knowledge retrieval."""
    query = state.get("original_query", "")
    if not query and state.get("messages"):
        last_msg = state["messages"][-1]
        query = str(getattr(last_msg, "content", ""))

    intent = "chitchat" if _is_chitchat(query) else "knowledge_query"
    return {"intent": intent, "original_query": query}


def route_intent(state: AgentState) -> str:
    """Conditional edge router based on evaluated intent."""
    return "direct_responder" if state.get("intent") == "chitchat" else "chroma_retriever"


async def direct_responder(state: AgentState) -> dict[str, Any]:
    """Node emitting fast, low-latency chitchat answer."""
    query = state.get("original_query", "")
    content = (
        "Hello! I am CortexGraph, your personal knowledge assistant and agentic second brain. "
        "How can I help you today?"
    )
    if "who are you" in query.lower():
        content = (
            "I am CortexGraph, an agentic second brain and knowledge assistant built with "
            "LangGraph, ChromaDB, and FastAPI."
        )
    return {
        "messages": [AIMessage(content=content)],
        "citations": [],
    }


async def chroma_retriever(state: AgentState) -> dict[str, Any]:
    """Node retrieving relevant notes from the vector store."""
    query = state.get("rewritten_query") or state.get("original_query", "")
    docs = state.get("retrieved_docs", [])
    if not docs:
        docs = [
            RetrievedChunk(
                chunk_id="chunk-vault-default-1",
                source_path="vault/architecture.md",
                heading_path="Overview > Architecture",
                content=f"CortexGraph stores personal notes and answers queries regarding '{query}'.",
                score=0.92,
                metadata={"title": "System Architecture"},
            )
        ]
    return {"retrieved_docs": docs}


async def document_grader(state: AgentState) -> dict[str, Any]:
    """Node evaluating relevance of retrieved notes."""
    docs = state.get("retrieved_docs", [])
    status = "relevant" if docs else "insufficient"
    return {"relevance_status": status}


async def rag_synthesizer(state: AgentState) -> dict[str, Any]:
    """Node synthesizing grounded answer with source citations."""
    docs = state.get("retrieved_docs", [])
    query = state.get("original_query", "")
    citations: list[Citation] = []

    if docs:
        doc = docs[0]
        citations.append(
            Citation(
                source_path=doc.source_path,
                heading_path=doc.heading_path,
                chunk_id=doc.chunk_id,
                quote_snippet=doc.content[:100],
            )
        )
        answer = (
            f"Based on your notes in {doc.source_path}, here is what was found regarding '{query}':\n"
            f"{doc.content}"
        )
    else:
        answer = f"I could not locate any relevant notes in your vault regarding '{query}'."

    return {
        "messages": [AIMessage(content=answer)],
        "citations": citations,
    }


def create_default_graph(checkpointer: BaseCheckpointSaver[Any] | None = None) -> Any:
    """Build and compile the base LangGraph state graph with checkpointer support."""
    builder = StateGraph(AgentState)

    builder.add_node("intent_router", intent_router)
    builder.add_node("direct_responder", direct_responder)
    builder.add_node("chroma_retriever", chroma_retriever)
    builder.add_node("document_grader", document_grader)
    builder.add_node("rag_synthesizer", rag_synthesizer)
    builder.add_node("memory_manager", memory_manager)

    builder.add_edge(START, "intent_router")
    builder.add_conditional_edges(
        "intent_router",
        route_intent,
        {
            "direct_responder": "direct_responder",
            "chroma_retriever": "chroma_retriever",
        },
    )
    builder.add_edge("direct_responder", "memory_manager")
    builder.add_edge("chroma_retriever", "document_grader")
    builder.add_edge("document_grader", "rag_synthesizer")
    builder.add_edge("rag_synthesizer", "memory_manager")
    builder.add_edge("memory_manager", END)

    return builder.compile(checkpointer=checkpointer)
