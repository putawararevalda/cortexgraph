"""Client wrapper for NVIDIA Auto-Ontology FastMCP / Streamable HTTP endpoint."""

from __future__ import annotations

import time
from typing import Any, cast

import structlog
from fastmcp import Client

from src.config import get_settings

logger = structlog.get_logger("cortexgraph.mcp")


def _extract_data(result: Any) -> Any:
    """Extract payload from a FastMCP tool call result."""
    if hasattr(result, "data"):
        return result.data
    return result


def _to_dict(payload: Any) -> dict[str, Any]:
    """Convert arbitrary tool payload into a dictionary."""
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "model_dump"):
        return cast(dict[str, Any], payload.model_dump())
    if hasattr(payload, "__dict__"):
        return dict(payload.__dict__)
    return {"raw": payload}


class AutoOntologyMcpClient:
    """Async client wrapper for NVIDIA Auto-Ontology MCP service."""

    def __init__(
        self,
        mcp_url: str | None = None,
        transport: Any | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the Auto-Ontology MCP client.

        Args:
            mcp_url: Configurable MCP HTTP endpoint URL (defaults to AUTO_ONTOLOGY_MCP_URL).
            transport: Optional custom FastMCP transport or in-memory server instance for testing.
            timeout: Tool execution timeout in seconds.
        """
        self.mcp_url = mcp_url or get_settings().auto_ontology_mcp_url
        self.transport = transport
        self.timeout = timeout

    def _create_session(self) -> Client[Any]:
        """Create a FastMCP Client session context."""
        if self.transport is not None:
            return Client(self.transport, timeout=self.timeout)
        return Client(self.mcp_url, timeout=self.timeout)

    async def check_answerable(self, query: str) -> bool:
        """Pre-flight check verifying if the semantic layer covers the entities in the query.

        Args:
            query: User's natural language question.

        Returns:
            True if entities in query have coverage in the semantic layer, False otherwise.
        """
        start_time = time.perf_counter()
        try:
            async with self._create_session() as session:
                result = await session.call_tool("check_answerable", {"question": query})
                data = _extract_data(result)
                latency_ms = (time.perf_counter() - start_time) * 1000

                coverage: float = 0.0
                if isinstance(data, dict):
                    coverage = float(data.get("coverage", 0.0))
                elif hasattr(data, "coverage"):
                    coverage = float(getattr(data, "coverage", 0.0))
                elif isinstance(data, (bool, int, float)):
                    coverage = float(data)

                is_covered = coverage > 0.0
                logger.info(
                    "mcp_check_answerable",
                    query=query,
                    coverage=coverage,
                    is_covered=is_covered,
                    mcp_latency_ms=round(latency_ms, 2),
                )
                return is_covered
        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(
                "mcp_check_answerable_failed",
                query=query,
                error=str(exc),
                mcp_latency_ms=round(latency_ms, 2),
            )
            return False

    async def ask_question(self, query: str) -> dict[str, Any]:
        """Execute text-to-SQL agent via Auto-Ontology MCP ask_question tool.

        Args:
            query: User's natural language database question.

        Returns:
            Dictionary containing 'answer', 'sql', 'rows', 'row_count', 'latency_ms', and optional 'error'.
        """
        start_time = time.perf_counter()
        try:
            async with self._create_session() as session:
                result = await session.call_tool("ask_question", {"question": query})
                data = _to_dict(_extract_data(result))
                latency_ms = (time.perf_counter() - start_time) * 1000

                sql = str(data.get("sql") or "")
                raw_rows = data.get("rows")
                rows: list[dict[str, Any]] = raw_rows if isinstance(raw_rows, list) else []
                row_count = int(data.get("row_count", len(rows)))

                logger.info(
                    "mcp_sql_execution",
                    query=query,
                    mcp_latency_ms=round(latency_ms, 2),
                    sql_generated=sql,
                    rows_returned=row_count,
                )
                return {
                    "answer": str(data.get("answer") or ""),
                    "sql": sql,
                    "rows": rows,
                    "row_count": row_count,
                    "truncated": bool(data.get("truncated", False)),
                    "reasoning": str(data.get("reasoning") or ""),
                    "latency_ms": round(latency_ms, 2),
                }
        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "mcp_ask_question_failed",
                query=query,
                error=str(exc),
                mcp_latency_ms=round(latency_ms, 2),
            )
            return {
                "answer": f"Unable to execute structured database query: {exc}",
                "sql": "",
                "rows": [],
                "row_count": 0,
                "truncated": False,
                "reasoning": "",
                "error": str(exc),
                "latency_ms": round(latency_ms, 2),
            }


_default_client: AutoOntologyMcpClient | None = None


def get_mcp_client(
    mcp_url: str | None = None,
    transport: Any | None = None,
) -> AutoOntologyMcpClient:
    """Retrieve or create an AutoOntologyMcpClient instance."""
    global _default_client
    if transport is not None or mcp_url is not None:
        return AutoOntologyMcpClient(mcp_url=mcp_url, transport=transport)
    if _default_client is None:
        _default_client = AutoOntologyMcpClient()
    return _default_client
