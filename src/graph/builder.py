"""StateGraph compilation and Mermaid visualization export for CortexGraph."""

import argparse
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.graph.edges import (
    route_after_grading,
    route_after_hallucination_check,
    route_by_intent,
)
from src.graph.nodes import (
    chroma_retriever,
    direct_responder,
    document_grader,
    hallucination_grader,
    intent_router,
    mcp_sql_node,
    memory_manager,
    query_analyzer,
    query_rewriter,
    rag_synthesizer,
)
from src.graph.state import AgentState


def build_graph(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    node_overrides: dict[str, Any] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile the bounded cyclic CortexGraph StateGraph with conditional edges and checkpointer.

    Topology:
      START -> intent_router
      intent_router -> (route_by_intent) ->
         - direct_responder -> memory_manager -> END
         - query_analyzer -> chroma_retriever -> document_grader
         - mcp_sql_node -> memory_manager -> END
      document_grader -> (route_after_grading: circuit breaker MAX_RETRIES = 2) ->
         - rag_synthesizer
         - query_rewriter -> chroma_retriever
      rag_synthesizer -> hallucination_grader
      hallucination_grader -> (route_after_hallucination_check: circuit breaker MAX_RETRIES = 2) ->
         - memory_manager -> END
         - rag_synthesizer
    """
    builder = StateGraph(AgentState)
    overrides = node_overrides or {}

    # Register nodes with override support
    builder.add_node("intent_router", overrides.get("intent_router", intent_router))
    builder.add_node("direct_responder", overrides.get("direct_responder", direct_responder))
    builder.add_node("query_analyzer", overrides.get("query_analyzer", query_analyzer))
    builder.add_node("chroma_retriever", overrides.get("chroma_retriever", chroma_retriever))
    builder.add_node("document_grader", overrides.get("document_grader", document_grader))
    builder.add_node("query_rewriter", overrides.get("query_rewriter", query_rewriter))
    builder.add_node("rag_synthesizer", overrides.get("rag_synthesizer", rag_synthesizer))
    builder.add_node("hallucination_grader", overrides.get("hallucination_grader", hallucination_grader))
    builder.add_node("memory_manager", overrides.get("memory_manager", memory_manager))
    builder.add_node("mcp_sql_node", overrides.get("mcp_sql_node", mcp_sql_node))

    # Flow wiring
    builder.add_edge(START, "intent_router")

    # Conditional 1: Intent Routing
    builder.add_conditional_edges(
        "intent_router",
        route_by_intent,
        {
            "query_analyzer": "query_analyzer",
            "direct_responder": "direct_responder",
            "mcp_sql_node": "mcp_sql_node",
        },
    )

    # Structured SQL branch
    builder.add_edge("mcp_sql_node", "memory_manager")

    # Direct chitchat branch
    builder.add_edge("direct_responder", "memory_manager")

    # Knowledge query RAG branch
    builder.add_edge("query_analyzer", "chroma_retriever")
    builder.add_edge("chroma_retriever", "document_grader")

    # Conditional 2: Grading / Query Rewrite Loop (circuit breaker: MAX_RETRIES = 2)
    builder.add_conditional_edges(
        "document_grader",
        route_after_grading,
        {
            "rag_synthesizer": "rag_synthesizer",
            "query_rewriter": "query_rewriter",
        },
    )

    # Query rewriter loops back to retriever
    builder.add_edge("query_rewriter", "chroma_retriever")

    # Synthesis & Hallucination check
    builder.add_edge("rag_synthesizer", "hallucination_grader")

    # Conditional 3: Hallucination Verification Loop (circuit breaker: MAX_RETRIES = 2)
    builder.add_conditional_edges(
        "hallucination_grader",
        route_after_hallucination_check,
        {
            "memory_manager": "memory_manager",
            "rag_synthesizer": "rag_synthesizer",
        },
    )

    # Conversational memory update to END
    builder.add_edge("memory_manager", END)

    return builder.compile(checkpointer=checkpointer)


def export_graph_diagram(output_path: str | Path | None = None) -> str:
    """Generate and optionally save the compiled Mermaid topology diagram for the README."""
    graph = build_graph()
    mermaid_diagram = graph.get_graph().draw_mermaid()

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(mermaid_diagram, encoding="utf-8")

    return mermaid_diagram


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export CortexGraph LangGraph Mermaid diagram.")
    parser.add_argument("--output", "-o", help="File path to save the Mermaid diagram", default=None)
    args = parser.parse_args()

    diagram = export_graph_diagram(output_path=args.output)
    if not args.output:
        print(diagram)
    else:
        print(f"Diagram successfully exported to {args.output}")
