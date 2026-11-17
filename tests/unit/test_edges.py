"""Unit tests for conditional edges, loop circuit breakers, and StateGraph compilation.

Verifies:
- All branches of route_by_intent, route_after_grading, and route_after_hallucination_check.
- Hard circuit breakers guaranteeing forced exit at MAX_RETRIES = 2.
- Zero possibility of infinite loops when graders permanently return 'insufficient' or 'hallucinated'.
- Turn-start counter resets inside intent_router.
- StateGraph builder compilation and Mermaid diagram export.
"""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.builder import build_graph, export_graph_diagram
from src.graph.edges import (
    MAX_RETRIES,
    route_after_grading,
    route_after_hallucination_check,
    route_by_intent,
)
from src.graph.nodes import intent_router
from src.graph.state import AgentState, create_initial_state


class TestRouteByIntent:
    """Tests verifying deterministic edge routing from intent_router."""

    def test_knowledge_query_routes_to_query_analyzer(self) -> None:
        state: AgentState = {"intent": "knowledge_query"}
        assert route_by_intent(state) == "query_analyzer"

    def test_structured_data_query_routes_to_mcp_sql_node(self) -> None:
        state: AgentState = {"intent": "structured_data_query"}
        assert route_by_intent(state) == "mcp_sql_node"

    def test_chitchat_routes_to_direct_responder(self) -> None:
        state: AgentState = {"intent": "chitchat"}
        assert route_by_intent(state) == "direct_responder"

    def test_clarify_routes_to_direct_responder(self) -> None:
        state: AgentState = {"intent": "clarify"}
        assert route_by_intent(state) == "direct_responder"

    def test_default_or_missing_intent_routes_to_direct_responder(self) -> None:
        empty_state: AgentState = {}
        assert route_by_intent(empty_state) == "direct_responder"

    def test_arbitrary_dict_input(self) -> None:
        raw_state: dict[str, Any] = {"intent": "knowledge_query", "extra": 123}
        assert route_by_intent(raw_state) == "query_analyzer"


class TestRouteAfterGrading:
    """Tests verifying conditional edge routing after document relevance grading."""

    def test_relevant_documents_route_to_rag_synthesizer(self) -> None:
        state: AgentState = {"relevance_status": "relevant", "rewrite_count": 0}
        assert route_after_grading(state) == "rag_synthesizer"

    def test_relevant_documents_at_first_retry_route_to_rag_synthesizer(self) -> None:
        state: AgentState = {"relevance_status": "relevant", "rewrite_count": 1}
        assert route_after_grading(state) == "rag_synthesizer"

    def test_insufficient_documents_at_count_0_routes_to_query_rewriter(self) -> None:
        state: AgentState = {"relevance_status": "insufficient", "rewrite_count": 0}
        assert route_after_grading(state) == "query_rewriter"

    def test_insufficient_documents_at_count_1_routes_to_query_rewriter(self) -> None:
        state: AgentState = {"relevance_status": "insufficient", "rewrite_count": 1}
        assert route_after_grading(state) == "query_rewriter"

    def test_circuit_breaker_forced_exit_at_max_retries(self) -> None:
        """When rewrite_count == MAX_RETRIES (2), must force exit to rag_synthesizer."""
        state: AgentState = {"relevance_status": "insufficient", "rewrite_count": 2}
        assert route_after_grading(state) == "rag_synthesizer"

    def test_circuit_breaker_forced_exit_exceeding_max_retries(self) -> None:
        """When rewrite_count > MAX_RETRIES (e.g. 3), must force exit to rag_synthesizer."""
        state: AgentState = {"relevance_status": "insufficient", "rewrite_count": 3}
        assert route_after_grading(state) == "rag_synthesizer"

    def test_missing_rewrite_count_defaults_to_zero(self) -> None:
        state: AgentState = {"relevance_status": "insufficient"}
        assert route_after_grading(state) == "query_rewriter"

    def test_missing_relevance_status_defaults_to_insufficient(self) -> None:
        state: AgentState = {"rewrite_count": 0}
        assert route_after_grading(state) == "query_rewriter"

    def test_missing_relevance_status_with_max_retries_exits(self) -> None:
        state: AgentState = {"rewrite_count": MAX_RETRIES}
        assert route_after_grading(state) == "rag_synthesizer"


class TestRouteAfterHallucinationCheck:
    """Tests verifying conditional edge routing after hallucination evaluation."""

    def test_grounded_answer_routes_to_memory_manager(self) -> None:
        state: dict[str, Any] = {"is_grounded": True, "generation_count": 0}
        assert route_after_hallucination_check(state) == "memory_manager"

    def test_grounded_status_routes_to_memory_manager(self) -> None:
        state: dict[str, Any] = {"hallucination_status": "grounded", "generation_count": 1}
        assert route_after_hallucination_check(state) == "memory_manager"

    def test_grounded_grade_routes_to_memory_manager(self) -> None:
        state: dict[str, Any] = {"hallucination_grade": "grounded", "generation_count": 0}
        assert route_after_hallucination_check(state) == "memory_manager"

    def test_hallucinated_answer_at_count_0_routes_to_rag_synthesizer(self) -> None:
        state: dict[str, Any] = {"is_grounded": False, "generation_count": 0}
        assert route_after_hallucination_check(state) == "rag_synthesizer"

    def test_hallucinated_answer_at_count_1_routes_to_rag_synthesizer(self) -> None:
        state: dict[str, Any] = {"is_grounded": False, "generation_count": 1}
        assert route_after_hallucination_check(state) == "rag_synthesizer"

    def test_ungrounded_status_at_count_0_routes_to_rag_synthesizer(self) -> None:
        state: dict[str, Any] = {"hallucination_status": "ungrounded", "generation_count": 0}
        assert route_after_hallucination_check(state) == "rag_synthesizer"

    def test_circuit_breaker_forced_exit_at_generation_max_retries(self) -> None:
        """When generation_count == MAX_RETRIES (2), must force exit to memory_manager."""
        state: dict[str, Any] = {"is_grounded": False, "generation_count": 2}
        assert route_after_hallucination_check(state) == "memory_manager"

    def test_circuit_breaker_forced_exit_exceeding_generation_max_retries(self) -> None:
        state: dict[str, Any] = {"is_grounded": False, "generation_count": 4}
        assert route_after_hallucination_check(state) == "memory_manager"

    def test_default_state_routes_to_memory_manager(self) -> None:
        state: dict[str, Any] = {"generation_count": 0}
        assert route_after_hallucination_check(state) == "memory_manager"


class TestLoopCircuitBreakerInfiniteLoopPrevention:
    """Verifies zero possibility of infinite loops with simulated permanent failures."""

    def test_query_rewrite_cycle_terminates_deterministically(self) -> None:
        """Simulate document_grader permanently returning insufficient.

        Loop must execute query_rewriter exactly MAX_RETRIES times (2)
        and then deterministically transition to rag_synthesizer.
        """
        state: AgentState = create_initial_state(
            original_query="Non-existent obscure concept",
            relevance_status="insufficient",
            rewrite_count=0,
        )

        loop_executions = 0
        path: list[str] = []

        while True:
            decision = route_after_grading(state)
            path.append(decision)

            if decision == "rag_synthesizer":
                break

            assert decision == "query_rewriter"
            state["rewrite_count"] = state.get("rewrite_count", 0) + 1
            loop_executions += 1

            # Safety guard in test to prevent test timeout if circuit breaker fails
            if loop_executions > 10:
                pytest.fail("Infinite loop detected: rewrite_count circuit breaker failed to trip!")

        assert loop_executions == MAX_RETRIES
        assert state["rewrite_count"] == MAX_RETRIES
        assert path == ["query_rewriter", "query_rewriter", "rag_synthesizer"]

    def test_hallucination_cycle_terminates_deterministically(self) -> None:
        """Simulate hallucination_grader permanently returning is_grounded = False.

        Loop must execute rag_synthesizer regeneration exactly MAX_RETRIES times (2)
        and then deterministically transition to memory_manager.
        """
        state: dict[str, Any] = {
            "is_grounded": False,
            "generation_count": 0,
        }

        loop_executions = 0
        path: list[str] = []

        while True:
            decision = route_after_hallucination_check(state)
            path.append(decision)

            if decision == "memory_manager":
                break

            assert decision == "rag_synthesizer"
            state["generation_count"] = state.get("generation_count", 0) + 1
            loop_executions += 1

            if loop_executions > 10:
                pytest.fail("Infinite loop detected: generation_count circuit breaker failed to trip!")

        assert loop_executions == MAX_RETRIES
        assert state["generation_count"] == MAX_RETRIES
        assert path == ["rag_synthesizer", "rag_synthesizer", "memory_manager"]

    def test_intent_router_resets_loop_counters_at_turn_start(self) -> None:
        """Verify that intent_router resets rewrite_count and generation_count to 0."""
        stale_state: AgentState = {
            "original_query": "What is CortexGraph?",
            "rewrite_count": 5,
            "generation_count": 3,
            "intent": "knowledge_query",
        }

        updated = intent_router(stale_state)
        assert updated["rewrite_count"] == 0
        assert updated["generation_count"] == 0
        assert updated["intent"] == "knowledge_query"
        assert updated["original_query"] == "What is CortexGraph?"


class TestCompiledStateGraphExecution:
    """Verifies end-to-end execution of the compiled StateGraph with bounded cycles."""

    @pytest.mark.asyncio
    async def test_compiled_graph_chitchat_execution(self) -> None:
        """START -> intent_router -> direct_responder -> memory_manager -> END."""
        graph = build_graph()
        initial_state = {
            "messages": [HumanMessage(content="Hello, who are you?", id="msg-1")],
            "original_query": "Hello, who are you?",
        }

        result = await graph.ainvoke(initial_state)
        assert result["intent"] == "chitchat"
        assert len(result["messages"]) >= 2
        last_msg = result["messages"][-1]
        assert "CortexGraph" in last_msg.content
        assert result["citations"] == []
        assert result["rewrite_count"] == 0
        assert result["generation_count"] == 0

    @pytest.mark.asyncio
    async def test_compiled_graph_rag_execution(self) -> None:
        """START -> intent_router -> query_analyzer -> chroma_retriever -> document_grader ->

        rag_synthesizer -> hallucination_grader -> memory_manager -> END.
        """
        def mock_grader(_s: Any) -> dict[str, Any]:
            return {"relevance_status": "relevant", "retrieved_docs": []}

        def mock_synthesizer(_s: Any) -> dict[str, Any]:
            return {
                "messages": [AIMessage(content="Architecture explanation [Source: notes/arch.md]")],
                "citations": [{"source_path": "notes/arch.md", "chunk_id": "c1", "heading_path": "", "quote_snippet": ""}],
                "generation_count": 0,
            }

        def mock_hallucination(_s: Any) -> dict[str, Any]:
            return {"is_grounded": True, "hallucination_grade": "grounded"}
        graph = build_graph(
            node_overrides={
                "document_grader": mock_grader,
                "rag_synthesizer": mock_synthesizer,
                "hallucination_grader": mock_hallucination,
            }
        )
        initial_state = {
            "messages": [HumanMessage(content="Explain architecture", id="msg-rag-1")],
            "original_query": "Explain architecture",
            "intent": "knowledge_query",
        }

        result = await graph.ainvoke(initial_state)
        assert result["intent"] == "knowledge_query"
        assert result["relevance_status"] == "relevant"
        assert len(result["citations"]) > 0
        assert result["rewrite_count"] == 0

    @pytest.mark.asyncio
    async def test_compiled_graph_structured_data_query_execution(self) -> None:
        """START -> intent_router -> mcp_sql_node -> memory_manager -> END."""
        from unittest.mock import AsyncMock

        mock_mcp = AsyncMock()
        mock_mcp.check_answerable.return_value = True
        mock_mcp.ask_question.return_value = {
            "answer": "Total sales count is 500.",
            "sql": "SELECT count(*) FROM sales.orders",
            "rows": [{"count": 500}],
            "row_count": 1,
            "latency_ms": 12.5,
        }

        async def custom_sql_node(state: AgentState) -> dict[str, Any]:
            from src.graph.nodes import mcp_sql_node

            return await mcp_sql_node(state, client=mock_mcp)

        graph = build_graph(node_overrides={"mcp_sql_node": custom_sql_node})
        initial_state = {
            "messages": [HumanMessage(content="What are the total sales?", id="msg-sql-1")],
            "original_query": "What are the total sales?",
            "intent": "structured_data_query",
        }

        result = await graph.ainvoke(initial_state)
        assert result["intent"] == "structured_data_query"
        assert len(result["messages"]) >= 2
        last_msg = result["messages"][-1]
        assert "Total sales count is 500" in last_msg.content
        assert "[Source: db://wideworldimporters#orders]" in last_msg.content
        assert result["sql_artifact"] is not None
        assert result["sql_artifact"].row_count == 1
        assert len(result["citations"]) == 1
        assert result["citations"][0].source_path == "db://wideworldimporters"

    @pytest.mark.asyncio
    async def test_compiled_graph_infinite_loop_prevention_mock_insufficient(self) -> None:
        """When document_grader permanently returns insufficient, graph must exit cleanly at MAX_RETRIES."""
        async def always_insufficient_grader(_state: AgentState) -> dict[str, Any]:
            return {"relevance_status": "insufficient"}

        graph = build_graph(
            node_overrides={"document_grader": always_insufficient_grader}
        )

        initial_state = {
            "messages": [HumanMessage(content="Search for nonexistent term", id="msg-fail-1")],
            "original_query": "Search for nonexistent term",
            "intent": "knowledge_query",
        }

        result = await graph.ainvoke(initial_state)
        # Should have rewritten exactly MAX_RETRIES (2) times and then forced exit to synthesizer
        assert result["rewrite_count"] == MAX_RETRIES
        assert len(result["messages"]) >= 2
        last_msg = result["messages"][-1]
        assert len(last_msg.content) > 0

    @pytest.mark.asyncio
    async def test_compiled_graph_infinite_loop_prevention_mock_hallucination(self) -> None:
        """When hallucination_grader permanently returns is_grounded = False, graph exits cleanly at MAX_RETRIES."""
        async def always_hallucinated_grader(_state: AgentState) -> dict[str, Any]:
            return {"is_grounded": False, "hallucination_status": "hallucinated"}

        graph = build_graph(
            node_overrides={"hallucination_grader": always_hallucinated_grader}
        )

        initial_state = {
            "messages": [HumanMessage(content="Generate answer with hallucination", id="msg-h-1")],
            "original_query": "Generate answer with hallucination",
            "intent": "knowledge_query",
        }

        result = await graph.ainvoke(initial_state)
        assert result["generation_count"] == MAX_RETRIES
        assert len(result["messages"]) >= 2


class TestMermaidDiagramExport:
    """Verifies Mermaid diagram export functionality."""

    def test_export_graph_diagram_contains_expected_nodes_and_edges(self) -> None:
        diagram = export_graph_diagram()
        assert isinstance(diagram, str)
        assert "graph TD" in diagram
        assert "intent_router" in diagram
        assert "direct_responder" in diagram
        assert "query_analyzer" in diagram
        assert "chroma_retriever" in diagram
        assert "document_grader" in diagram
        assert "query_rewriter" in diagram
        assert "rag_synthesizer" in diagram
        assert "hallucination_grader" in diagram
        assert "memory_manager" in diagram
        assert "mcp_sql_node" in diagram
        # Verify cycles exist in diagram text
        assert "query_rewriter --> chroma_retriever" in diagram
        assert "rag_synthesizer --> hallucination_grader" in diagram

    def test_export_graph_diagram_to_file(self, tmp_path: Any) -> None:
        out_file = tmp_path / "diagram.mmd"
        diagram = export_graph_diagram(output_path=out_file)
        assert out_file.exists()
        assert out_file.read_text(encoding="utf-8") == diagram
