"""Langfuse client singleton, LangGraph CallbackHandler instrumentation, prompt caching, and user feedback scoring."""

import logging
import os
import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig

from src.config import get_settings

logger = logging.getLogger(__name__)

# In-memory prompt cache: (prompt_name, label) -> (cached_at_timestamp, prompt_content)
_PROMPT_CACHE: dict[tuple[str, str], tuple[float, str]] = {}
CACHE_TTL_SECONDS = 300  # 5-minute TTL


class NoOpCallbackHandler(BaseCallbackHandler):
    """Graceful no-op callback handler used when Langfuse credentials are not configured."""

    pass


def is_langfuse_configured() -> bool:
    """Check whether Langfuse public and secret keys are configured in settings or environment."""
    settings = get_settings()
    pub = (settings.langfuse_public_key or os.environ.get("LANGFUSE_PUBLIC_KEY", "")).strip()
    sec = (settings.langfuse_secret_key or os.environ.get("LANGFUSE_SECRET_KEY", "")).strip()
    return bool(pub and sec)


_client_instance: Any = None


def get_langfuse_client(force_reload: bool = False) -> Any:
    """Retrieve the cached Langfuse client singleton.

    Returns the Langfuse client if configured, or None / disabled client if offline.
    """
    global _client_instance
    if _client_instance is not None and not force_reload:
        return _client_instance

    settings = get_settings()
    pub = (settings.langfuse_public_key or os.environ.get("LANGFUSE_PUBLIC_KEY", "")).strip()
    sec = (settings.langfuse_secret_key or os.environ.get("LANGFUSE_SECRET_KEY", "")).strip()
    host = settings.langfuse_host or os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

    try:
        from langfuse import Langfuse

        client = (
            Langfuse(
                public_key=pub,
                secret_key=sec,
                host=host,
            )
            if pub and sec
            else Langfuse()
        )

        # Alias score to create_score if score method is missing (Langfuse v5.x compatibility)
        if hasattr(client, "create_score") and not hasattr(client, "score"):
            client.score = client.create_score  # type: ignore[attr-defined]

        _client_instance = client
        return _client_instance
    except Exception as exc:
        logger.warning("Could not initialize Langfuse client: %s", exc)
        return None


def set_langfuse_client(client: Any) -> None:
    """Override singleton client instance (primarily for unit tests)."""
    global _client_instance
    _client_instance = client


def clear_prompt_cache() -> None:
    """Clear in-memory prompt cache."""
    _PROMPT_CACHE.clear()


def get_langfuse_runnable_config(
    thread_id: str,
    user_id: str | None,
    trace_id: str,
    phase_tag: str,
) -> RunnableConfig:
    """Create a LangGraph RunnableConfig instrumented with Langfuse tracing.

    Configures session grouping by thread_id, user tracking, phase tags,
    and attaches the Langfuse CallbackHandler. Degrades gracefully to
    a NoOpCallbackHandler if Langfuse credentials are not configured.

    Args:
        thread_id: Conversation thread identifier (Langfuse session ID).
        user_id: User identifier or None (defaults to 'anonymous').
        trace_id: Execution trace identifier used as run_id and trace_id.
        phase_tag: Operational phase tag (e.g., 'chat', 'chat_stream', 'eval').

    Returns:
        RunnableConfig dictionary ready for LangGraph invocation.
    """
    settings = get_settings()
    configured = is_langfuse_configured()

    callbacks: list[Any] = []
    if configured:
        try:
            from langfuse.langchain import CallbackHandler

            pub_key = (
                settings.langfuse_public_key or os.environ.get("LANGFUSE_PUBLIC_KEY", "")
            ).strip()
            handler = CallbackHandler(
                public_key=pub_key,
                trace_context={"trace_id": trace_id},
            )
            callbacks.append(handler)
        except Exception as exc:
            logger.warning(
                "Failed to instantiate Langfuse CallbackHandler (%s); using no-op fallback.", exc
            )
            callbacks.append(NoOpCallbackHandler())
    else:
        callbacks.append(NoOpCallbackHandler())

    run_id_val: Any = trace_id
    config: RunnableConfig = {
        "callbacks": callbacks,
        "run_id": run_id_val,
        "metadata": {
            "langfuse_session_id": thread_id,
            "langfuse_user_id": user_id or "anonymous",
            "langfuse_tags": ["cortexgraph", phase_tag],
        },
        "configurable": {
            "thread_id": thread_id,
        },
        "tags": ["cortexgraph", phase_tag],
    }
    return config


def _get_local_prompt_fallback(prompt_name: str) -> str:
    """Retrieve fallback system template string from local constants in src.rag.prompts."""
    from src.rag import prompts as rag_prompts

    # 1. Check PROMPT_REGISTRY directly
    if hasattr(rag_prompts, "PROMPT_REGISTRY"):
        registry = rag_prompts.PROMPT_REGISTRY
        if prompt_name in registry:
            return str(registry[prompt_name].system_template)
        # Check normalized key (remove PROMPT_ prefix, lower-case)
        normalized = prompt_name.lower().removeprefix("prompt_")
        if normalized in registry:
            return str(registry[normalized].system_template)

    # 2. Check module attributes (e.g. PROMPT_INTENT_ROUTER)
    attr_candidates = [
        prompt_name,
        prompt_name.upper(),
        f"PROMPT_{prompt_name.upper()}",
    ]
    for attr in attr_candidates:
        if hasattr(rag_prompts, attr):
            target = getattr(rag_prompts, attr)
            if hasattr(target, "system_template"):
                return str(target.system_template)
            if isinstance(target, str):
                return target

    raise KeyError(
        f"Prompt '{prompt_name}' not found in Langfuse or local registry in src.rag.prompts"
    )


def get_prompt(prompt_name: str, label: str = "production", **kwargs: Any) -> str:
    """Fetch active prompt version from Langfuse Prompt Management with local fallback.

    Caches fetched prompt in memory for 5 minutes (TTL = 300 seconds).
    If Langfuse is unconfigured, unreachable, or returns an error, seamlessly
    falls back to the local prompt definitions in `src/rag/prompts.py`.

    Args:
        prompt_name: Name of the prompt in Langfuse / local registry.
        label: Langfuse deployment label (defaults to 'production').
        **kwargs: Optional fallback or version overrides.

    Returns:
        System prompt template string.
    """
    now = time.time()
    cache_key = (prompt_name, label)

    # 1. Return from in-memory cache if still valid within 5 minutes
    if cache_key in _PROMPT_CACHE:
        cached_time, cached_content = _PROMPT_CACHE[cache_key]
        if now - cached_time < CACHE_TTL_SECONDS:
            return cached_content

    # 2. If manual fallback override passed in kwargs, use it
    if "fallback" in kwargs and kwargs["fallback"] is not None:
        return str(kwargs["fallback"])

    # 3. Attempt remote fetch from Langfuse Prompt Management
    if is_langfuse_configured():
        try:
            client = get_langfuse_client()
            if client is not None and hasattr(client, "get_prompt"):
                version = kwargs.get("version")
                remote_prompt = client.get_prompt(
                    name=prompt_name,
                    label=label if version is None else None,
                    version=version,
                )
                if remote_prompt is not None:
                    prompt_text = ""
                    if hasattr(remote_prompt, "prompt"):
                        prompt_text = str(remote_prompt.prompt)
                    elif hasattr(remote_prompt, "compile"):
                        prompt_text = str(remote_prompt.compile())
                    else:
                        prompt_text = str(remote_prompt)

                    if prompt_text:
                        _PROMPT_CACHE[cache_key] = (now, prompt_text)
                        return prompt_text
        except Exception as exc:
            logger.debug(
                "Langfuse prompt fetch for '%s' (label='%s') failed (%s), using local fallback.",
                prompt_name,
                label,
                exc,
            )

    # 4. Fall back seamlessly to local constants in src.rag.prompts
    fallback_prompt = _get_local_prompt_fallback(prompt_name)
    _PROMPT_CACHE[cache_key] = (now, fallback_prompt)
    return fallback_prompt


class _LangfuseProxy:
    """Proxy object providing top-level langfuse.score(...) and delegating client methods."""

    def score(
        self,
        trace_id: str,
        name: str,
        value: int | float,
        comment: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Record score against a trace ID via the active Langfuse client."""
        client = get_langfuse_client()
        if client is not None:
            if hasattr(client, "score"):
                return client.score(
                    trace_id=trace_id, name=name, value=value, comment=comment, **kwargs
                )
            if hasattr(client, "create_score"):
                return client.create_score(
                    trace_id=trace_id, name=name, value=value, comment=comment, **kwargs
                )
        logger.debug(
            "Langfuse client is offline; skipped scoring trace_id=%s, name=%s, value=%s",
            trace_id,
            name,
            value,
        )
        return None

    def get_prompt(self, name: str, label: str = "production", **kwargs: Any) -> Any:
        client = get_langfuse_client()
        if client is not None and hasattr(client, "get_prompt"):
            return client.get_prompt(name=name, label=label, **kwargs)
        return None

    def __getattr__(self, item: str) -> Any:
        client = get_langfuse_client()
        if client is not None:
            return getattr(client, item)
        raise AttributeError(f"Langfuse client is not initialized and has no attribute '{item}'")


# Singleton proxy instance providing langfuse.score(...)
langfuse = _LangfuseProxy()


def record_feedback(trace_id: str, score: int, comment: str | None = None) -> bool:
    """Record user feedback score (0 or 1) against a specific Langfuse trace.

    Args:
        trace_id: The target Langfuse trace ID.
        score: Feedback rating, must be strictly 0 or 1.
        comment: Optional textual explanation or user feedback.

    Returns:
        True if the score call was dispatched, False otherwise.
    """
    if score not in (0, 1):
        raise ValueError(f"Feedback score must be strictly 0 or 1, got {score}")

    try:
        langfuse.score(
            trace_id=trace_id,
            name="user_feedback",
            value=score,
            comment=comment,
        )
        return True
    except Exception as exc:
        logger.error(
            "Failed to record feedback score for trace %s: %s", trace_id, exc, exc_info=True
        )
        return False
