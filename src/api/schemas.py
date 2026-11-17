"""Pydantic schemas for CortexGraph FastAPI communication."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Citation(BaseModel):
    """Inline source citation linked to retrieved knowledge chunks."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    source: str = Field(default="", description="Source document path or identifier")
    source_path: str = Field(default="", description="Source document path or identifier")
    heading_path: str = Field(default="", description="Markdown heading path inside document")
    chunk_id: str = Field(default="", description="Unique chunk identifier")
    content: str = Field(default="", description="Referenced excerpt or snippet text")
    quote_snippet: str = Field(default="", description="Referenced excerpt or snippet text")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary citation metadata"
    )

    @model_validator(mode="before")
    @classmethod
    def sync_aliases(cls, data: Any) -> Any:
        """Ensure source and content are bidirectional with source_path and quote_snippet."""
        if isinstance(data, dict):
            # Sync source / source_path
            src = data.get("source") or data.get("source_path") or ""
            data["source"] = src
            data["source_path"] = src

            # Sync content / quote_snippet
            txt = data.get("content") or data.get("quote_snippet") or ""
            data["content"] = txt
            data["quote_snippet"] = txt
        return data


class ChatRequest(BaseModel):
    """Payload schema for chat interaction requests."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    message: str = Field(..., min_length=1, description="User query or message text", alias="query")
    thread_id: str | None = Field(
        default=None, description="Conversation thread identifier for memory persistence"
    )
    metadata_filters: dict[str, Any] | None = Field(
        default=None, description="Optional metadata filter parameters (tags, dates, source types)"
    )

    @model_validator(mode="before")
    @classmethod
    def sync_query(cls, data: Any) -> Any:
        """Allow query as alias for message."""
        if isinstance(data, dict):
            msg = data.get("message") or data.get("query")
            if msg is not None:
                data["message"] = msg
        return data


class ChatResponse(BaseModel):
    """Payload schema for synchronous chat responses."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    thread_id: str = Field(..., description="Conversation thread identifier")
    response: str = Field(..., description="Assistant response text")
    citations: list[Citation] = Field(
        default_factory=list, description="Extracted document citations"
    )
    run_id: str | None = Field(default=None, description="Execution run identifier")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Execution metadata")

    @model_validator(mode="before")
    @classmethod
    def sync_response(cls, data: Any) -> Any:
        """Allow answer as input alias for response."""
        if isinstance(data, dict):
            ans = data.get("answer") or data.get("response")
            if ans is not None:
                data["response"] = ans
        return data


StreamEventType = Literal["node_status", "token", "citations", "done", "error"]


class StreamEvent(BaseModel):
    """Structured event emitted across Server-Sent Events (SSE) stream."""

    model_config = ConfigDict(extra="ignore")

    type: StreamEventType = Field(..., description="SSE event type discriminator")
    node: str | None = Field(default=None, description="Graph node name for status events")
    message: str | None = Field(
        default=None, description="Human friendly status text or error message"
    )
    content: str | None = Field(default=None, description="Streamed token string")
    citations: list[Citation] | None = Field(default=None, description="Document citations payload")
    thread_id: str | None = Field(default=None, description="Conversation thread identifier")
    run_id: str | None = Field(default=None, description="Execution run identifier")


class ThreadMessage(BaseModel):
    """Individual message turn in thread history."""

    model_config = ConfigDict(extra="ignore")

    role: str = Field(..., description="Role of message sender (user, assistant, system)")
    content: str = Field(..., description="Text content of message")
    created_at: str | None = Field(default=None, description="Timestamp in ISO 8601 format")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Message metadata")


class ThreadHistoryResponse(BaseModel):
    """Payload schema for thread conversation history."""

    model_config = ConfigDict(extra="ignore")

    thread_id: str = Field(..., description="Conversation thread identifier")
    messages: list[ThreadMessage] = Field(
        default_factory=list, description="Ordered conversation turns"
    )
    summary: str | None = Field(default=None, description="Rolling conversation summary")


class FeedbackRequest(BaseModel):
    """Payload schema for submitting user feedback on an agent trace."""

    model_config = ConfigDict(extra="ignore")

    trace_id: str = Field(..., min_length=1, description="Langfuse trace identifier to score")
    score: int = Field(..., ge=0, le=1, description="Binary user feedback rating (0 or 1)")
    comment: str | None = Field(
        default=None, description="Optional textual user feedback or rationale"
    )


class FeedbackResponse(BaseModel):
    """Response schema for user feedback submission."""

    model_config = ConfigDict(extra="ignore")

    status: str = Field(default="ok", description="Status of the feedback submission")
    trace_id: str = Field(..., description="Target trace identifier")
    score: int = Field(..., description="Recorded score value")
    comment: str | None = Field(default=None, description="Recorded user comment")

