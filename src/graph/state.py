"""State schema definitions and models for CortexGraph."""

from typing import Annotated, Any, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

__all__ = [
    "AgentState",
    "RetrievedChunk",
    "Citation",
    "SqlArtifact",
    "create_initial_state",
]


class RetrievedChunk(BaseModel):
    """A retrieved knowledge chunk from the vector store with source metadata."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    chunk_id: str
    content: str
    source_path: str
    heading_path: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """Inline source citation linked to retrieved document chunks."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    source_path: str
    heading_path: str = ""
    chunk_id: str
    quote_snippet: str = ""


class SqlArtifact(BaseModel):
    """Execution artifact capturing structured SQL query, preview rows, and execution latency."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    query: str
    generated_sql: str
    row_count: int
    data_preview: list[dict[str, Any]] = Field(default_factory=list)
    execution_time_ms: float = 0.0


class AgentState(TypedDict, total=False):
    """The core state dictionary for the CortexGraph LangGraph agent."""

    messages: Annotated[list[AnyMessage], add_messages]
    summary: str
    intent: Literal["chitchat", "knowledge_query", "structured_data_query", "clarify"]
    original_query: str
    rewritten_query: str
    metadata_filters: dict[str, Any]
    retrieved_docs: list[RetrievedChunk]
    relevance_status: Literal["relevant", "insufficient"]
    rewrite_count: int
    generation_count: int
    citations: list[Citation]
    is_grounded: bool
    hallucination_critique: str | None
    sql_artifact: SqlArtifact | None


def create_initial_state(
    original_query: str = "",
    messages: list[AnyMessage] | None = None,
    summary: str = "",
    intent: Literal["chitchat", "knowledge_query", "structured_data_query", "clarify"] = "chitchat",
    rewritten_query: str = "",
    metadata_filters: dict[str, Any] | None = None,
    retrieved_docs: list[RetrievedChunk] | None = None,
    relevance_status: Literal["relevant", "insufficient"] = "insufficient",
    rewrite_count: int = 0,
    generation_count: int = 0,
    citations: list[Citation] | None = None,
    is_grounded: bool = True,
    hallucination_critique: str | None = None,
    sql_artifact: SqlArtifact | None = None,
) -> AgentState:
    """Helper to initialize an AgentState dict with standard defaults."""
    return AgentState(
        messages=messages or [],
        summary=summary,
        intent=intent,
        original_query=original_query,
        rewritten_query=rewritten_query,
        metadata_filters=metadata_filters or {},
        retrieved_docs=retrieved_docs or [],
        relevance_status=relevance_status,
        rewrite_count=rewrite_count,
        generation_count=generation_count,
        citations=citations or [],
        is_grounded=is_grounded,
        hallucination_critique=hallucination_critique,
        sql_artifact=sql_artifact,
    )
