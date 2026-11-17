"""ASGI middleware for correlation context propagation and structlog contextvars binding."""

import uuid
from typing import Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


def get_correlation_context() -> dict[str, Any]:
    """Retrieve active correlation identifiers from structlog contextvars."""
    return structlog.contextvars.get_contextvars()


def bind_correlation_context(
    request_id: str | None = None,
    trace_id: str | None = None,
    thread_id: str | None = None,
    user_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Explicitly bind or update correlation keys in structlog contextvars."""
    updates: dict[str, Any] = {}
    if request_id is not None:
        updates["request_id"] = request_id
    if trace_id is not None:
        updates["trace_id"] = trace_id
    if thread_id is not None:
        updates["thread_id"] = thread_id
    if user_id is not None:
        updates["user_id"] = user_id
    updates.update(extra)

    if updates:
        structlog.contextvars.bind_contextvars(**updates)
    return structlog.contextvars.get_contextvars()


class CorrelationMiddleware(BaseHTTPMiddleware):
    """ASGI correlation middleware binding request correlation IDs into structlog contextvars.

    Extracts `X-Request-Id`, `X-Trace-Id`, `X-Thread-Id`, and `X-User-Id` from incoming headers
    or query parameters, binds them to `structlog.contextvars`, sets them on `request.state`,
    and sets them on the outgoing HTTP response headers.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        # 1. Resolve request_id and trace_id
        req_id = (
            request.headers.get("X-Request-Id")
            or request.headers.get("x-request-id")
            or uuid.uuid4().hex
        )
        trace_id = (
            request.headers.get("X-Trace-Id")
            or request.headers.get("x-trace-id")
            or uuid.uuid4().hex
        )

        # 2. Resolve thread_id and user_id from headers or query parameters
        thread_id = (
            request.headers.get("X-Thread-Id")
            or request.headers.get("x-thread-id")
            or request.query_params.get("thread_id")
        )
        user_id = (
            request.headers.get("X-User-Id")
            or request.headers.get("x-user-id")
            or request.query_params.get("user_id")
        )

        # 3. Attach to request.state for downstream handler access
        request.state.request_id = req_id
        request.state.trace_id = trace_id
        request.state.thread_id = thread_id
        request.state.user_id = user_id

        # 4. Clean and bind into structlog contextvars
        structlog.contextvars.clear_contextvars()
        bindings: dict[str, Any] = {
            "request_id": req_id,
            "trace_id": trace_id,
        }
        if thread_id:
            bindings["thread_id"] = thread_id
        if user_id:
            bindings["user_id"] = user_id

        structlog.contextvars.bind_contextvars(**bindings)

        # 5. Invoke downstream application pipeline
        response = await call_next(request)

        # 6. Set response correlation headers
        response.headers["X-Request-Id"] = req_id
        response.headers["X-Trace-Id"] = trace_id
        if thread_id and "X-Thread-Id" not in response.headers:
            response.headers["X-Thread-Id"] = thread_id
        if user_id and "X-User-Id" not in response.headers:
            response.headers["X-User-Id"] = user_id

        return response


# Alias for backward and forward naming compatibility
ASGICorrelationMiddleware = CorrelationMiddleware


async def correlation_id_middleware(
    request: Request,
    call_next: RequestResponseEndpoint,
) -> Response:
    """HTTP middleware handler compatible with @app.middleware('http')."""
    req_id = (
        request.headers.get("X-Request-Id")
        or request.headers.get("x-request-id")
        or uuid.uuid4().hex
    )
    trace_id = (
        request.headers.get("X-Trace-Id")
        or request.headers.get("x-trace-id")
        or uuid.uuid4().hex
    )
    thread_id = (
        request.headers.get("X-Thread-Id")
        or request.headers.get("x-thread-id")
        or request.query_params.get("thread_id")
    )
    user_id = (
        request.headers.get("X-User-Id")
        or request.headers.get("x-user-id")
        or request.query_params.get("user_id")
    )

    request.state.request_id = req_id
    request.state.trace_id = trace_id
    request.state.thread_id = thread_id
    request.state.user_id = user_id

    structlog.contextvars.clear_contextvars()
    bindings: dict[str, Any] = {
        "request_id": req_id,
        "trace_id": trace_id,
    }
    if thread_id:
        bindings["thread_id"] = thread_id
    if user_id:
        bindings["user_id"] = user_id
    structlog.contextvars.bind_contextvars(**bindings)

    response = await call_next(request)

    response.headers["X-Request-Id"] = req_id
    response.headers["X-Trace-Id"] = trace_id
    if thread_id and "X-Thread-Id" not in response.headers:
        response.headers["X-Thread-Id"] = thread_id
    if user_id and "X-User-Id" not in response.headers:
        response.headers["X-User-Id"] = user_id

    return response


__all__ = [
    "CorrelationMiddleware",
    "ASGICorrelationMiddleware",
    "correlation_id_middleware",
    "bind_correlation_context",
    "get_correlation_context",
]
