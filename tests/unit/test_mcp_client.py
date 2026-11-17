"""Unit tests for AutoOntologyMcpClient with FastMCP in-memory server and error mocks."""

from typing import Any

import pytest
from fastmcp import FastMCP

from src.rag.mcp_client import AutoOntologyMcpClient, get_mcp_client


@pytest.fixture
def in_memory_mcp_server() -> FastMCP:
    """Fixture providing an in-memory FastMCP server with Auto-Ontology tools."""
    server = FastMCP(name="test-auto-ontology")

    @server.tool(name="check_answerable")
    def mock_check_answerable(question: str) -> dict[str, Any]:
        if "unanswerable" in question.lower() or "weather" in question.lower():
            return {"coverage": 0.0, "candidates": []}
        return {"coverage": 0.85, "candidates": [{"entity": "customers"}]}

    @server.tool(name="ask_question")
    def mock_ask_question(question: str) -> dict[str, Any]:
        if "fail" in question.lower():
            raise RuntimeError("Database query execution failed")
        return {
            "answer": "Wide World Importers has 663 customers.",
            "sql": 'SELECT count(*) FROM "SALES"."CUSTOMERS"',
            "rows": [{"count": 663}],
            "row_count": 1,
            "truncated": False,
            "reasoning": "Identified customers table and performed count",
        }

    return server


class TestAutoOntologyMcpClient:
    """Tests verifying AutoOntologyMcpClient pre-flight and execution methods."""

    @pytest.mark.asyncio
    async def test_check_answerable_true(self, in_memory_mcp_server: FastMCP) -> None:
        client = AutoOntologyMcpClient(transport=in_memory_mcp_server)
        is_answerable = await client.check_answerable("How many customers do we have?")
        assert is_answerable is True

    @pytest.mark.asyncio
    async def test_check_answerable_false_when_uncovered(self, in_memory_mcp_server: FastMCP) -> None:
        client = AutoOntologyMcpClient(transport=in_memory_mcp_server)
        is_answerable = await client.check_answerable("What is the weather in Tokyo unanswerable?")
        assert is_answerable is False

    @pytest.mark.asyncio
    async def test_check_answerable_graceful_on_transport_error(self) -> None:
        # Invalid URL where no server is listening
        client = AutoOntologyMcpClient(mcp_url="http://127.0.0.1:59999/mcp", timeout=0.5)
        is_answerable = await client.check_answerable("Any question")
        assert is_answerable is False

    @pytest.mark.asyncio
    async def test_ask_question_success(self, in_memory_mcp_server: FastMCP) -> None:
        client = AutoOntologyMcpClient(transport=in_memory_mcp_server)
        result = await client.ask_question("How many customers do we have?")

        assert "663 customers" in result["answer"]
        assert 'SELECT count(*) FROM "SALES"."CUSTOMERS"' in result["sql"]
        assert result["rows"] == [{"count": 663}]
        assert result["row_count"] == 1
        assert result["truncated"] is False
        assert result["latency_ms"] >= 0.0

    @pytest.mark.asyncio
    async def test_ask_question_error_handling(self, in_memory_mcp_server: FastMCP) -> None:
        client = AutoOntologyMcpClient(transport=in_memory_mcp_server)
        result = await client.ask_question("Please fail this query")

        assert "error" in result or "Unable to execute" in result["answer"]
        assert result["rows"] == []
        assert result["row_count"] == 0

    def test_get_mcp_client_singleton(self) -> None:
        client1 = get_mcp_client()
        client2 = get_mcp_client()
        assert client1 is client2
        assert client1.mcp_url.startswith("http://")
