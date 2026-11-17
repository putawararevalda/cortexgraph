"""Tenacity retry decorators and backoff resiliency for OpenAI API rate limits and network errors."""

from collections.abc import Callable
from typing import Any, TypeVar

import openai
import structlog
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

logger = structlog.get_logger("cortexgraph.resiliency")

# Default retryable OpenAI exceptions (RateLimitError for HTTP 429, APIConnectionError for network drops)
DEFAULT_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
)

F = TypeVar("F", bound=Callable[..., Any])


def log_retry_attempt(retry_state: RetryCallState) -> None:
    """Log a structured warning on every retry attempt before backing off."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    attempt_num = retry_state.attempt_number
    sleep_seconds = (
        round(retry_state.next_action.sleep, 2)
        if retry_state.next_action is not None
        else 0.0
    )

    logger.warning(
        "openai_retry_attempt",
        attempt_number=attempt_num,
        sleep_seconds=sleep_seconds,
        error_type=type(exc).__name__ if exc else "Unknown",
        error_message=str(exc) if exc else "",
    )


def retry_openai_call(
    fn: F | None = None,
    *,
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    retry_exceptions: tuple[type[Exception], ...] | None = None,
) -> Any:
    """Decorator providing exponential jitter backoff for OpenAI API rate limits and transient errors.

    Catches OpenAI RateLimitError (HTTP 429) and APIConnectionError, retrying up to max_attempts
    with wait_random_exponential(min=min_wait, max=max_wait) and logging structured
    event='openai_retry_attempt' warning before each sleep.

    Can be used either with or without parameters:
        @retry_openai_call
        def call_openai(): ...

        @retry_openai_call(max_attempts=3, min_wait=1, max_wait=10)
        async def call_openai_async(): ...
    """
    target_exceptions = retry_exceptions or DEFAULT_RETRYABLE_EXCEPTIONS

    decorator = retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(min=min_wait, max=max_wait),
        retry=retry_if_exception_type(target_exceptions),
        before_sleep=log_retry_attempt,
    )

    if fn is not None:
        return decorator(fn)
    return decorator


__all__ = [
    "retry_openai_call",
    "log_retry_attempt",
    "DEFAULT_RETRYABLE_EXCEPTIONS",
]
