"""CortexGraph FastAPI REST and SSE endpoints."""

from src.api.main import app, create_app
from src.api.middleware import (
    ASGICorrelationMiddleware,
    CorrelationMiddleware,
    bind_correlation_context,
    correlation_id_middleware,
    get_correlation_context,
)
from src.api.routes_ingest import IngestRequest, IngestResponse
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    FeedbackRequest,
    FeedbackResponse,
    StreamEvent,
    ThreadHistoryResponse,
    ThreadMessage,
)

__all__ = [
    "app",
    "create_app",
    "CorrelationMiddleware",
    "ASGICorrelationMiddleware",
    "correlation_id_middleware",
    "bind_correlation_context",
    "get_correlation_context",
    "ChatRequest",
    "ChatResponse",
    "Citation",
    "FeedbackRequest",
    "FeedbackResponse",
    "StreamEvent",
    "ThreadHistoryResponse",
    "ThreadMessage",
    "IngestRequest",
    "IngestResponse",
]
