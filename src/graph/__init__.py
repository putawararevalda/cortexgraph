"""CortexGraph graph package: state schema, checkpointer factory, nodes, edges, and builder."""

from typing import Any

__all__ = [
    "AgentState",
    "RetrievedChunk",
    "Citation",
    "SqlArtifact",
    "create_initial_state",
    "get_checkpointer",
    "create_sqlite_checkpointer",
    "create_postgres_checkpointer",
    "verify_thread_id_config",
    "build_graph",
    "export_graph_diagram",
    "MAX_RETRIES",
    "route_by_intent",
    "route_after_grading",
    "route_after_hallucination_check",
    "intent_router",
    "direct_responder",
    "query_analyzer",
    "chroma_retriever",
    "document_grader",
    "query_rewriter",
    "rag_synthesizer",
    "hallucination_grader",
    "memory_manager",
    "mcp_sql_node",
    "count_message_tokens",
]


def __getattr__(name: str) -> Any:
    if name in (
        "AgentState",
        "RetrievedChunk",
        "Citation",
        "SqlArtifact",
        "create_initial_state",
    ):
        from src.graph import state

        return getattr(state, name)

    if name in (
        "get_checkpointer",
        "create_sqlite_checkpointer",
        "create_postgres_checkpointer",
        "verify_thread_id_config",
    ):
        from src.graph import checkpointer

        return getattr(checkpointer, name)

    if name in (
        "build_graph",
        "export_graph_diagram",
    ):
        from src.graph import builder

        return getattr(builder, name)

    if name in (
        "MAX_RETRIES",
        "route_by_intent",
        "route_after_grading",
        "route_after_hallucination_check",
    ):
        from src.graph import edges

        return getattr(edges, name)

    if name in (
        "intent_router",
        "direct_responder",
        "query_analyzer",
        "chroma_retriever",
        "document_grader",
        "query_rewriter",
        "rag_synthesizer",
        "hallucination_grader",
        "memory_manager",
        "mcp_sql_node",
        "count_message_tokens",
    ):
        from src.graph import nodes

        return getattr(nodes, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
