"""Integration tests for end-to-end Auto-Ontology MCP SQL query graph flow.

Verifies:
- State transitions through `intent_router -> mcp_sql_node -> memory_manager -> END`.
- Deterministic offline execution with simulated structured OpenAI classifications
  and recorded Wide World Importers MCP payloads.
- Correct population of `SqlArtifact` in AgentState (query, generated_sql, row_count, data_preview).
- Strict citation generation matching `[Source: db://wideworldimporters#<table>]` and Citation models.
- Graceful error handling and fallback responses when MCP server is unreachable or unanswerable.
- Execution completes offline in < 5 seconds.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.graph.builder import build_graph
from src.graph.nodes import IntentClassification
from src.graph.state import Citation, SqlArtifact, create_initial_state

# ============================================================================
# Deterministic Recorded Payloads for Wide World Importers (Offline)
# ============================================================================

WIDE_WORLD_CUSTOMERS_PAYLOAD = {
    "answer": (
        "The top 5 customers by sales volume in Wide World Importers are Tailspin Toys "
        "(Head Office), Wingtip Toys (Head Office), Tailspin Toys (Sylvanite), "
        "Wingtip Toys (Rifle), and Tailspin Toys (Pecan Grove)."
    ),
    "sql": (
        "SELECT c.CustomerName, SUM(il.ExtendedPrice - il.TaxAmount) AS TotalNetSales "
        "FROM sales.Invoices i "
        "JOIN sales.InvoiceLines il ON i.InvoiceID = il.InvoiceID "
        "JOIN sales.Customers c ON i.CustomerID = c.CustomerID "
        "GROUP BY c.CustomerName "
        "ORDER BY TotalNetSales DESC "
        "LIMIT 5;"
    ),
    "rows": [
        {"CustomerName": "Tailspin Toys (Head Office)", "TotalNetSales": 482100.50},
        {"CustomerName": "Wingtip Toys (Head Office)", "TotalNetSales": 394200.00},
        {"CustomerName": "Tailspin Toys (Sylvanite)", "TotalNetSales": 281550.75},
        {"CustomerName": "Wingtip Toys (Rifle)", "TotalNetSales": 265400.25},
        {"CustomerName": "Tailspin Toys (Pecan Grove)", "TotalNetSales": 241900.00},
    ],
    "row_count": 5,
    "truncated": False,
    "reasoning": "Aggregated sales.InvoiceLines joined with sales.Customers.",
    "latency_ms": 12.4,
}

WIDE_WORLD_TOTAL_SALES_PAYLOAD = {
    "answer": "Total net sales across all fiscal periods amounts to $14,892,450.00.",
    "sql": "SELECT SUM(ExtendedPrice - TaxAmount) AS TotalSales FROM sales.InvoiceLines;",
    "rows": [{"TotalSales": 14892450.00}],
    "row_count": 1,
    "truncated": False,
    "reasoning": "Summed ExtendedPrice minus TaxAmount from sales.InvoiceLines.",
    "latency_ms": 8.1,
}

WIDE_WORLD_STOCK_PAYLOAD = {
    "answer": "There are currently 1,840 units of 'USB-C Charging Hub 7-Port' in stock.",
    "sql": (
        "SELECT s.StockItemName, h.QuantityOnHand "
        "FROM warehouse.StockItems s "
        "JOIN warehouse.StockItemHoldings h ON s.StockItemID = h.StockItemID "
        "WHERE s.StockItemName LIKE '%USB-C Charging Hub%';"
    ),
    "rows": [{"StockItemName": "USB-C Charging Hub 7-Port", "QuantityOnHand": 1840}],
    "row_count": 1,
    "truncated": False,
    "reasoning": "Queried warehouse.StockItemHoldings joined with warehouse.StockItems.",
    "latency_ms": 9.5,
}


class MockAutoOntologyClient:
    """Deterministic offline mock of the Auto-Ontology FastMCP client."""

    def __init__(
        self,
        is_answerable: bool = True,
        payload: dict[str, Any] | None = None,
        should_fail: bool = False,
        error_message: str = "Connection refused to Auto-Ontology MCP service",
    ) -> None:
        self._is_answerable = is_answerable
        self._payload = payload or WIDE_WORLD_CUSTOMERS_PAYLOAD
        self._should_fail = should_fail
        self._error_message = error_message

    async def check_answerable(self, _query: str) -> bool:
        if self._should_fail:
            raise ConnectionError(self._error_message)
        return self._is_answerable

    async def ask_question(self, _query: str) -> dict[str, Any]:
        if self._should_fail:
            return {
                "answer": f"Unable to execute structured database query: {self._error_message}",
                "sql": "",
                "rows": [],
                "row_count": 0,
                "error": self._error_message,
                "latency_ms": 0.0,
            }
        return dict(self._payload)


def _build_mock_router_llm(intent: str = "structured_data_query") -> MagicMock:
    """Build mock ChatOpenAI structured output LLM for intent routing."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_structured.invoke.return_value = IntentClassification(
        intent=intent,  # type: ignore[arg-type]
        reasoning="Query asks for structured relational database metrics from Wide World Importers.",
    )
    return mock_llm


# ============================================================================
# Integration Tests
# ============================================================================


class TestSqlGraphFlowIntegration:
    """End-to-end integration tests verifying the full StateGraph SQL flow."""

    @pytest.mark.asyncio
    async def test_successful_sql_graph_flow_top_customers(self) -> None:
        """Verify full path: intent_router -> mcp_sql_node -> memory_manager -> END.

        Tests top customers query with SQL generation, tabular preview, and source citation.
        """
        mock_client = MockAutoOntologyClient(payload=WIDE_WORLD_CUSTOMERS_PAYLOAD)
        mock_llm = _build_mock_router_llm(intent="structured_data_query")

        # Custom node adapter injecting mock_llm into intent_router and mock_client into mcp_sql_node
        from src.graph.nodes import intent_router, mcp_sql_node

        def custom_intent_router(state: Any) -> Any:
            return intent_router(state, llm=mock_llm)

        async def custom_mcp_sql_node(state: Any) -> Any:
            return await mcp_sql_node(state, client=mock_client)

        checkpointer = MemorySaver()
        graph = build_graph(
            checkpointer=checkpointer,
            node_overrides={
                "intent_router": custom_intent_router,
                "mcp_sql_node": custom_mcp_sql_node,
            },
        )

        query = "What are the top 5 customers by sales volume in Wide World Importers?"
        initial_state = create_initial_state(
            original_query=query,
            messages=[HumanMessage(content=query)],
        )

        config = {"configurable": {"thread_id": "test-thread-sql-flow-001"}}
        final_state = await graph.ainvoke(initial_state, config=config)

        # 1. State intent & routing verification
        assert final_state["intent"] == "structured_data_query"

        # 2. SqlArtifact populated with execution metrics & rows
        sql_artifact: SqlArtifact | None = final_state.get("sql_artifact")
        assert sql_artifact is not None
        assert isinstance(sql_artifact, SqlArtifact)
        assert sql_artifact.query == query
        assert "sales.Customers" in sql_artifact.generated_sql
        assert sql_artifact.row_count == 5
        assert len(sql_artifact.data_preview) == 5
        assert sql_artifact.data_preview[0]["CustomerName"] == "Tailspin Toys (Head Office)"
        assert sql_artifact.execution_time_ms >= 0

        # 3. Citation generation verification
        citations: list[Citation] = final_state.get("citations", [])
        assert len(citations) >= 1
        citation = citations[0]
        assert citation.source_path == "db://wideworldimporters"
        assert citation.heading_path in ("Customers", "Invoices")
        assert citation.chunk_id == "sql_execution"
        assert "sales.Invoices" in citation.quote_snippet or "sales.Customers" in citation.quote_snippet

        # 4. Final Assistant response verification
        messages = final_state["messages"]
        assert len(messages) >= 2
        last_message = messages[-1]
        assert isinstance(last_message, AIMessage)
        assert "Tailspin Toys" in last_message.content
        assert f"[Source: db://wideworldimporters#{citation.heading_path}]" in last_message.content

    @pytest.mark.asyncio
    async def test_successful_sql_graph_flow_total_sales(self) -> None:
        """Verify total sales query flows through SQL branch and formats citation."""
        mock_client = MockAutoOntologyClient(payload=WIDE_WORLD_TOTAL_SALES_PAYLOAD)
        mock_llm = _build_mock_router_llm(intent="structured_data_query")

        from src.graph.nodes import intent_router, mcp_sql_node

        def custom_intent_router(state: Any) -> Any:
            return intent_router(state, llm=mock_llm)

        async def custom_mcp_sql_node(state: Any) -> Any:
            return await mcp_sql_node(state, client=mock_client)

        graph = build_graph(
            checkpointer=MemorySaver(),
            node_overrides={
                "intent_router": custom_intent_router,
                "mcp_sql_node": custom_mcp_sql_node,
            },
        )

        query = "What is our total sales revenue for Wide World Importers?"
        initial_state = create_initial_state(
            original_query=query,
            messages=[HumanMessage(content=query)],
        )

        config = {"configurable": {"thread_id": "test-thread-sql-sales-002"}}
        final_state = await graph.ainvoke(initial_state, config=config)

        assert final_state["intent"] == "structured_data_query"
        sql_artifact = final_state.get("sql_artifact")
        assert sql_artifact is not None
        assert sql_artifact.row_count == 1
        assert sql_artifact.data_preview[0]["TotalSales"] == 14892450.00

        citations = final_state.get("citations", [])
        assert len(citations) == 1
        assert citations[0].heading_path == "InvoiceLines"

        last_msg = final_state["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert "[Source: db://wideworldimporters#InvoiceLines]" in last_msg.content

    @pytest.mark.asyncio
    async def test_successful_sql_graph_flow_stock_quantities(self) -> None:
        """Verify warehouse inventory count query flows through SQL branch."""
        mock_client = MockAutoOntologyClient(payload=WIDE_WORLD_STOCK_PAYLOAD)
        mock_llm = _build_mock_router_llm(intent="structured_data_query")

        from src.graph.nodes import intent_router, mcp_sql_node

        def custom_intent_router(state: Any) -> Any:
            return intent_router(state, llm=mock_llm)

        async def custom_mcp_sql_node(state: Any) -> Any:
            return await mcp_sql_node(state, client=mock_client)

        graph = build_graph(
            checkpointer=MemorySaver(),
            node_overrides={
                "intent_router": custom_intent_router,
                "mcp_sql_node": custom_mcp_sql_node,
            },
        )

        query = "How many units of USB-C Charging Hub are in stock?"
        initial_state = create_initial_state(
            original_query=query,
            messages=[HumanMessage(content=query)],
        )

        config = {"configurable": {"thread_id": "test-thread-sql-stock-003"}}
        final_state = await graph.ainvoke(initial_state, config=config)

        assert final_state["intent"] == "structured_data_query"
        sql_artifact = final_state.get("sql_artifact")
        assert sql_artifact is not None
        assert sql_artifact.data_preview[0]["QuantityOnHand"] == 1840

        last_msg = final_state["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert "[Source: db://wideworldimporters#StockItems]" in last_msg.content

    @pytest.mark.asyncio
    async def test_unanswerable_query_fallback(self) -> None:
        """Verify graceful fallback when query is not covered by semantic layer."""
        mock_client = MockAutoOntologyClient(is_answerable=False)
        mock_llm = _build_mock_router_llm(intent="structured_data_query")

        from src.graph.nodes import intent_router, mcp_sql_node

        def custom_intent_router(state: Any) -> Any:
            return intent_router(state, llm=mock_llm)

        async def custom_mcp_sql_node(state: Any) -> Any:
            return await mcp_sql_node(state, client=mock_client)

        graph = build_graph(
            checkpointer=MemorySaver(),
            node_overrides={
                "intent_router": custom_intent_router,
                "mcp_sql_node": custom_mcp_sql_node,
            },
        )

        query = "What is the secret recipe for grandmother's apple pie?"
        initial_state = create_initial_state(
            original_query=query,
            messages=[HumanMessage(content=query)],
        )

        config = {"configurable": {"thread_id": "test-thread-sql-unanswerable-004"}}
        final_state = await graph.ainvoke(initial_state, config=config)

        assert final_state["intent"] == "structured_data_query"
        assert final_state.get("sql_artifact") is None
        assert final_state.get("citations") == []

        last_msg = final_state["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert "not covered by the current semantic layer" in last_msg.content

    @pytest.mark.asyncio
    async def test_unreachable_mcp_server_fallback(self) -> None:
        """Verify graceful handling when MCP server is down or returns connection error."""
        mock_client = MockAutoOntologyClient(should_fail=True)
        mock_llm = _build_mock_router_llm(intent="structured_data_query")

        from src.graph.nodes import intent_router, mcp_sql_node

        def custom_intent_router(state: Any) -> Any:
            return intent_router(state, llm=mock_llm)

        async def custom_mcp_sql_node(state: Any) -> Any:
            return await mcp_sql_node(state, client=mock_client)

        graph = build_graph(
            checkpointer=MemorySaver(),
            node_overrides={
                "intent_router": custom_intent_router,
                "mcp_sql_node": custom_mcp_sql_node,
            },
        )

        query = "List all customer accounts with outstanding balances"
        initial_state = create_initial_state(
            original_query=query,
            messages=[HumanMessage(content=query)],
        )

        config = {"configurable": {"thread_id": "test-thread-sql-unreachable-005"}}
        final_state = await graph.ainvoke(initial_state, config=config)

        assert final_state["intent"] == "structured_data_query"
        assert final_state.get("sql_artifact") is None
        assert final_state.get("citations") == []

        last_msg = final_state["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert "Connection refused" in last_msg.content or "Unable to" in last_msg.content
