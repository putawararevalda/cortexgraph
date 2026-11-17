"""Deterministic conditional edge routing functions and loop circuit breakers for CortexGraph."""

from typing import Any

from src.graph.state import AgentState

MAX_RETRIES: int = 2


def route_by_intent(state: AgentState | dict[str, Any]) -> str:
    """Route execution flow from intent_router to appropriate subsystem.

    - "knowledge_query" -> transitions to "query_analyzer"
    - "structured_data_query" -> transitions to "mcp_sql_node"
    - "chitchat" or "clarify" -> transitions to "direct_responder"
    """
    intent = state.get("intent", "chitchat")
    if intent == "knowledge_query":
        return "query_analyzer"
    if intent == "structured_data_query":
        return "mcp_sql_node"
    return "direct_responder"


def route_after_grading(state: AgentState | dict[str, Any]) -> str:
    """Route execution flow after document relevance grading with hard circuit breaker.

    - Hard circuit breaker: if rewrite_count >= MAX_RETRIES (2), forced exit to "rag_synthesizer".
    - If relevance_status == "relevant", transition to "rag_synthesizer".
    - If relevance_status == "insufficient" and rewrite_count < MAX_RETRIES, transition to "query_rewriter".
    """
    rewrite_count = state.get("rewrite_count", 0)
    if rewrite_count >= MAX_RETRIES:
        return "rag_synthesizer"

    relevance_status = state.get("relevance_status", "insufficient")
    if relevance_status == "relevant":
        return "rag_synthesizer"

    return "query_rewriter"


def route_after_hallucination_check(state: AgentState | dict[str, Any]) -> str:
    """Route execution flow after hallucination verification with hard circuit breaker.

    - Hard circuit breaker: if generation_count >= MAX_RETRIES (2), forced exit to "memory_manager".
    - If answer is grounded, transition to "memory_manager".
    - If answer is hallucinated/ungrounded and generation_count < MAX_RETRIES, loop back to "rag_synthesizer".
    """
    generation_count = state.get("generation_count", 0)
    if generation_count >= MAX_RETRIES:
        return "memory_manager"

    is_grounded = state.get("is_grounded")
    hallucination_status = state.get("hallucination_status")
    hallucination_grade = state.get("hallucination_grade")

    is_hallucinated = (
        is_grounded is False
        or hallucination_status in ("hallucinated", "ungrounded", "insufficient")
        or hallucination_grade in ("hallucinated", "ungrounded")
    )

    if is_hallucinated:
        return "rag_synthesizer"

    return "memory_manager"
