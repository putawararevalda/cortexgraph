"""Structured logging configuration, zero-leak secret redaction, and node timing telemetry."""

import functools
import inspect
import logging
import os
import re
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from typing import Any, TypeVar, cast

import structlog
from structlog.types import EventDict, WrappedLogger

# Sensitive key names to automatically scrub from event dicts
SENSITIVE_KEY_NAMES: frozenset[str] = frozenset(
    {
        "api_key",
        "authorization",
        "openai_api_key",
        "langfuse_secret_key",
        "langfuse_public_key",
        "token",
        "secret",
        "password",
    }
)

# Regex matching OpenAI, Langfuse, and standard secret key formats
SECRET_PATTERN: re.Pattern[str] = re.compile(
    r"(?:sk|pk-lf|sk-lf)-[A-Za-z0-9_\-\.]+",
    re.IGNORECASE,
)

F = TypeVar("F", bound=Callable[..., Any])


def is_sensitive_key(key: Any) -> bool:
    """Determine if a key name indicates sensitive or credential content."""
    k_lower = str(key).lower()
    return (
        k_lower in SENSITIVE_KEY_NAMES
        or "api_key" in k_lower
        or "secret_key" in k_lower
        or "authorization" in k_lower
        or k_lower.startswith(("sk-", "pk-lf-", "sk-lf-"))
    )


def redact_value(val: Any) -> Any:
    """Recursively scrub secrets from strings, dictionaries, lists, tuples, and sets."""
    if isinstance(val, str):
        if SECRET_PATTERN.fullmatch(val.strip()):
            return "[REDACTED]"
        if SECRET_PATTERN.search(val):
            return SECRET_PATTERN.sub("[REDACTED]", val)
        return val
    elif isinstance(val, dict):
        return {
            k: ("[REDACTED]" if is_sensitive_key(k) else redact_value(v))
            for k, v in val.items()
        }
    elif isinstance(val, list):
        return [redact_value(item) for item in val]
    elif isinstance(val, tuple):
        return tuple(redact_value(item) for item in val)
    elif isinstance(val, set):
        return {redact_value(item) for item in val}
    return val


def redact_secrets_processor(
    _logger: WrappedLogger,
    _method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Structlog processor that masks sensitive credentials with [REDACTED]."""
    cleaned: EventDict = {}
    for key, val in event_dict.items():
        if is_sensitive_key(key):
            cleaned[key] = "[REDACTED]"
        else:
            cleaned[key] = redact_value(val)
    return cleaned


def setup_logging(
    log_format: str | None = None,
    log_level: str | None = None,
) -> None:
    """Configure structlog and standard library logging pipeline.

    Args:
        log_format: Format switch ('json' for single-line JSON, 'console' for colorized dev output).
        log_level: Minimum log level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').
    """
    format_choice = (
        log_format
        or os.environ.get("LOG_FORMAT")
        or "console"
    ).lower()

    level_choice = (
        log_level
        or os.environ.get("LOG_LEVEL")
        or "INFO"
    ).upper()

    numeric_level = getattr(logging, level_choice, logging.INFO)

    # Common shared processors before rendering (excluding filter_by_level for foreign_pre_chain)
    common_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        redact_secrets_processor,
    ]

    structlog_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        redact_secrets_processor,
    ]

    if format_choice == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    # Configure stdlib logging bridge
    stdlib_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=common_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Reset existing root handlers to prevent duplicated logs
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(stdlib_formatter)
    stream_handler.setLevel(numeric_level)
    root_logger.addHandler(stream_handler)

    # Configure structlog
    structlog.configure(
        processors=[
            *structlog_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Retrieve a configured structlog bound logger."""
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name or "cortexgraph"))


def log_node_execution(node_name: str) -> Callable[[F], F]:
    """Decorator for LangGraph node functions logging lifecycle, duration, and state delta.

    Supports both synchronous and asynchronous node functions.
    Emits event='node_started' and event='node_completed' with:
      - node_name
      - duration_ms (rounded to 2 decimal places)
      - rewrite_count
      - updated_state_keys (list of keys returned by the node)
      - correlation keys (request_id, thread_id, trace_id) merged via structlog.contextvars
    """
    def decorator(func: F) -> F:
        logger = get_logger("cortexgraph.nodes")

        def _extract_execution_context(
            args: Sequence[Any],
            kwargs: dict[str, Any],
        ) -> tuple[int, dict[str, Any]]:
            """Extract state and rewrite count, and ensure correlation context exists."""
            state: Any = args[0] if args and isinstance(args[0], dict) else kwargs.get("state")
            rewrite_count = 0
            if isinstance(state, dict):
                rewrite_count = int(state.get("rewrite_count") or 0)

            # Ensure correlation keys exist in contextvars for isolated test invocations
            ctx = structlog.contextvars.get_contextvars()
            bind_updates: dict[str, Any] = {}
            if "request_id" not in ctx:
                bind_updates["request_id"] = uuid.uuid4().hex
            if "trace_id" not in ctx:
                bind_updates["trace_id"] = uuid.uuid4().hex
            if "thread_id" not in ctx:
                t_id = (
                    (state.get("thread_id") if isinstance(state, dict) else None)
                    or kwargs.get("config", {}).get("configurable", {}).get("thread_id")
                    or "default-thread"
                )
                bind_updates["thread_id"] = str(t_id)

            if bind_updates:
                structlog.contextvars.bind_contextvars(**bind_updates)

            return rewrite_count, state

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                rewrite_count, _ = _extract_execution_context(args, kwargs)

                logger.info(
                    "node_started",
                    node_name=node_name,
                    rewrite_count=rewrite_count,
                )

                start_time = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                except Exception as exc:
                    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
                    logger.error(
                        "node_failed",
                        node_name=node_name,
                        duration_ms=duration_ms,
                        error=str(exc),
                    )
                    raise

                duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
                updated_keys: list[str] = list(result.keys()) if isinstance(result, dict) else []
                final_rewrite_count = (
                    int(result.get("rewrite_count", rewrite_count))
                    if isinstance(result, dict)
                    else rewrite_count
                )

                logger.info(
                    "node_completed",
                    node_name=node_name,
                    duration_ms=duration_ms,
                    rewrite_count=final_rewrite_count,
                    updated_state_keys=updated_keys,
                    updated_keys=updated_keys,
                )
                return result

            return async_wrapper  # type: ignore[return-value]
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                rewrite_count, _ = _extract_execution_context(args, kwargs)

                logger.info(
                    "node_started",
                    node_name=node_name,
                    rewrite_count=rewrite_count,
                )

                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                except Exception as exc:
                    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
                    logger.error(
                        "node_failed",
                        node_name=node_name,
                        duration_ms=duration_ms,
                        error=str(exc),
                    )
                    raise

                duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
                updated_keys: list[str] = list(result.keys()) if isinstance(result, dict) else []
                final_rewrite_count = (
                    int(result.get("rewrite_count", rewrite_count))
                    if isinstance(result, dict)
                    else rewrite_count
                )

                logger.info(
                    "node_completed",
                    node_name=node_name,
                    duration_ms=duration_ms,
                    rewrite_count=final_rewrite_count,
                    updated_state_keys=updated_keys,
                    updated_keys=updated_keys,
                )
                return result

            return sync_wrapper  # type: ignore[return-value]

    return decorator


__all__ = [
    "setup_logging",
    "get_logger",
    "log_node_execution",
    "redact_secrets_processor",
    "redact_value",
    "is_sensitive_key",
    "SECRET_PATTERN",
    "SENSITIVE_KEY_NAMES",
]

