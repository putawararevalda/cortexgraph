"""Unit tests for Langfuse LLM observability, LangGraph callbacks, prompt caching, and feedback API."""

import os
import time
from collections.abc import AsyncIterator, Iterator
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import create_app
from src.observability.langfuse_client import (
    CACHE_TTL_SECONDS,
    NoOpCallbackHandler,
    clear_prompt_cache,
    get_langfuse_runnable_config,
    get_prompt,
    langfuse,
    record_feedback,
    set_langfuse_client,
)
from src.rag.prompts import PROMPT_REGISTRY


@pytest.fixture(autouse=True)
def reset_langfuse_state() -> Iterator[None]:
    """Reset prompt cache and client overrides before and after each test."""
    clear_prompt_cache()
    set_langfuse_client(None)
    yield
    clear_prompt_cache()
    set_langfuse_client(None)


@pytest.fixture
async def api_client() -> AsyncIterator[AsyncClient]:
    """AsyncClient fixture with application lifespan context initialized."""
    test_app = create_app()
    transport = ASGITransport(app=test_app)
    async with (
        test_app.router.lifespan_context(test_app),
        AsyncClient(transport=transport, base_url="http://testserver") as client,
    ):
        yield client


class TestLangfuseRunnableConfig:
    """Tests for get_langfuse_runnable_config and graceful degradation."""

    def test_offline_graceful_degradation_to_noop_callback(self) -> None:
        """When Langfuse keys are absent, returns NoOpCallbackHandler and exact metadata."""
        with patch("src.observability.langfuse_client.is_langfuse_configured", return_value=False):
            config = get_langfuse_runnable_config(
                thread_id="thread-xyz",
                user_id=None,
                trace_id="trace-12345",
                phase_tag="chat",
            )

            assert "callbacks" in config
            callbacks = config["callbacks"]
            assert isinstance(callbacks, list)
            assert len(callbacks) == 1
            assert isinstance(callbacks[0], NoOpCallbackHandler)
            assert config["run_id"] == "trace-12345"
            assert config["metadata"]["langfuse_session_id"] == "thread-xyz"
            assert config["metadata"]["langfuse_user_id"] == "anonymous"
            assert config["metadata"]["langfuse_tags"] == ["cortexgraph", "chat"]
            assert config["configurable"]["thread_id"] == "thread-xyz"
            assert config["tags"] == ["cortexgraph", "chat"]

    def test_custom_user_id_propagation(self) -> None:
        """Verify user_id is recorded in metadata when provided."""
        with patch("src.observability.langfuse_client.is_langfuse_configured", return_value=False):
            config = get_langfuse_runnable_config(
                thread_id="thread-abc",
                user_id="user-revalda",
                trace_id="trace-67890",
                phase_tag="eval",
            )
            assert config["metadata"]["langfuse_user_id"] == "user-revalda"
            assert config["metadata"]["langfuse_tags"] == ["cortexgraph", "eval"]

    def test_configured_langfuse_instantiates_callback_handler(self) -> None:
        """When keys are configured, instantiates official CallbackHandler with trace_context."""
        mock_handler = MagicMock()
        with (
            patch(
                "src.observability.langfuse_client.is_langfuse_configured",
                return_value=True,
            ),
            patch("src.observability.langfuse_client.get_settings") as mock_settings,
            patch.dict(os.environ, {"LANGFUSE_PUBLIC_KEY": "pk-lf-test"}),
            patch("langfuse.langchain.CallbackHandler", return_value=mock_handler) as mock_cls,
        ):
            mock_settings.return_value.langfuse_public_key = "pk-lf-test"
            config = get_langfuse_runnable_config(
                thread_id="thread-real",
                user_id="user-1",
                trace_id="trace-real",
                phase_tag="chat_stream",
            )

            mock_cls.assert_called_once_with(
                public_key="pk-lf-test",
                trace_context={"trace_id": "trace-real"},
            )
            assert config["callbacks"] == [mock_handler]
            assert config["run_id"] == "trace-real"
            assert config["metadata"]["langfuse_session_id"] == "thread-real"

    def test_handler_instantiation_failure_falls_back_to_noop(self) -> None:
        """If CallbackHandler raises an exception, gracefully falls back to NoOpCallbackHandler."""
        with (
            patch(
                "src.observability.langfuse_client.is_langfuse_configured",
                return_value=True,
            ),
            patch(
                "langfuse.langchain.CallbackHandler",
                side_effect=Exception("Connection refused"),
            ),
        ):
            config = get_langfuse_runnable_config(
                thread_id="t1",
                user_id=None,
                trace_id="tr1",
                phase_tag="chat",
            )
            callbacks = config["callbacks"]
            assert isinstance(callbacks, list)
            assert len(callbacks) == 1
            assert isinstance(callbacks[0], NoOpCallbackHandler)


class TestPromptManagementAndCaching:
    """Tests for prompt fetching, in-memory TTL caching, and local fallback."""

    def test_local_fallback_all_seven_prompts(self) -> None:
        """Verify seamless local fallback for all registered prompts when Langfuse is offline."""
        for prompt_name, definition in PROMPT_REGISTRY.items():
            prompt = get_prompt(prompt_name)
            assert prompt == definition.system_template
            assert len(prompt) > 40

    def test_remote_prompt_fetching_and_ttl_caching(self) -> None:
        """Remote prompt is fetched and cached in memory with 5-minute TTL."""
        mock_client = MagicMock()
        mock_remote_prompt = MagicMock()
        mock_remote_prompt.prompt = "Remote prompt version 2.0"
        mock_client.get_prompt.return_value = mock_remote_prompt

        with (
            patch(
                "src.observability.langfuse_client.is_langfuse_configured",
                return_value=True,
            ),
            patch(
                "src.observability.langfuse_client.get_langfuse_client",
                return_value=mock_client,
            ),
        ):
            # First fetch: calls remote Langfuse client
            prompt1 = get_prompt("intent_router", label="production")
            assert prompt1 == "Remote prompt version 2.0"
            assert mock_client.get_prompt.call_count == 1

            # Second fetch within 5-minute TTL: served from cache without calling client again
            prompt2 = get_prompt("intent_router", label="production")
            assert prompt2 == "Remote prompt version 2.0"
            assert mock_client.get_prompt.call_count == 1

    def test_cache_expiration_refetches(self) -> None:
        """Cache entry older than 300 seconds triggers a fresh fetch."""
        mock_client = MagicMock()
        mock_remote_1 = MagicMock(prompt="Remote v1")
        mock_remote_2 = MagicMock(prompt="Remote v2")
        mock_client.get_prompt.side_effect = [mock_remote_1, mock_remote_2]

        with (
            patch(
                "src.observability.langfuse_client.is_langfuse_configured",
                return_value=True,
            ),
            patch(
                "src.observability.langfuse_client.get_langfuse_client",
                return_value=mock_client,
            ),
        ):
            # First fetch
            p1 = get_prompt("query_analyzer")
            assert p1 == "Remote v1"
            assert mock_client.get_prompt.call_count == 1

            # Simulate expiration by manipulating cache timestamp
            cache_key = ("query_analyzer", "production")
            from src.observability.langfuse_client import _PROMPT_CACHE

            _PROMPT_CACHE[cache_key] = (time.time() - (CACHE_TTL_SECONDS + 10), "Remote v1")

            # Second fetch after expiration: triggers refetch
            p2 = get_prompt("query_analyzer")
            assert p2 == "Remote v2"
            assert mock_client.get_prompt.call_count == 2

    def test_remote_failure_falls_back_to_local_prompts(self) -> None:
        """If Langfuse throws an error, fallback to local prompt definition seamlessly."""
        mock_client = MagicMock()
        mock_client.get_prompt.side_effect = ConnectionError("Langfuse cloud down")

        with (
            patch(
                "src.observability.langfuse_client.is_langfuse_configured",
                return_value=True,
            ),
            patch(
                "src.observability.langfuse_client.get_langfuse_client",
                return_value=mock_client,
            ),
        ):
            prompt = get_prompt("rag_synthesizer")
            assert "Mandatory Inline Citations" in prompt
            assert "Abstention Guardrail" in prompt

    def test_unknown_prompt_raises_key_error(self) -> None:
        """Unknown prompt name with no local definition raises KeyError."""
        with pytest.raises(KeyError):
            get_prompt("non_existent_custom_prompt_xyz")


class TestUserFeedbackAndScoring:
    """Tests for record_feedback and POST /api/v1/feedback endpoint."""

    def test_record_feedback_dispatches_langfuse_score(self) -> None:
        """record_feedback invokes langfuse.score with trace_id, user_feedback, and value."""
        with patch.object(langfuse, "score") as mock_score:
            result = record_feedback(
                trace_id="tr-abc-123",
                score=1,
                comment="Great citation accuracy!",
            )
            assert result is True
            mock_score.assert_called_once_with(
                trace_id="tr-abc-123",
                name="user_feedback",
                value=1,
                comment="Great citation accuracy!",
            )

    def test_record_feedback_negative_rating(self) -> None:
        """Negative rating 0 is recorded properly."""
        with patch.object(langfuse, "score") as mock_score:
            result = record_feedback(
                trace_id="tr-fail-456",
                score=0,
                comment="Hallucinated information.",
            )
            assert result is True
            mock_score.assert_called_once_with(
                trace_id="tr-fail-456",
                name="user_feedback",
                value=0,
                comment="Hallucinated information.",
            )

    def test_record_feedback_invalid_score_raises_value_error(self) -> None:
        """Scores other than 0 or 1 raise ValueError."""
        with pytest.raises(ValueError, match="strictly 0 or 1"):
            record_feedback(trace_id="tr-1", score=2)

        with pytest.raises(ValueError, match="strictly 0 or 1"):
            record_feedback(trace_id="tr-1", score=-1)

    @pytest.mark.asyncio
    async def test_feedback_api_positive_score(self, api_client: AsyncClient) -> None:
        """POST /api/v1/feedback attaches positive rating to trace_id."""
        with patch.object(langfuse, "score") as mock_score:
            payload = {
                "trace_id": "test-trace-999",
                "score": 1,
                "comment": "Accurate and well grounded response.",
            }
            response = await api_client.post("/api/v1/feedback", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["trace_id"] == "test-trace-999"
            assert data["score"] == 1
            assert data["comment"] == "Accurate and well grounded response."

            mock_score.assert_called_once_with(
                trace_id="test-trace-999",
                name="user_feedback",
                value=1,
                comment="Accurate and well grounded response.",
            )

    @pytest.mark.asyncio
    async def test_feedback_api_negative_score_without_comment(
        self, api_client: AsyncClient
    ) -> None:
        """POST /api/v1/feedback attaches negative rating without comment."""
        with patch.object(langfuse, "score") as mock_score:
            payload = {
                "trace_id": "test-trace-000",
                "score": 0,
            }
            response = await api_client.post("/api/v1/feedback", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["trace_id"] == "test-trace-000"
            assert data["score"] == 0
            assert data["comment"] is None

            mock_score.assert_called_once_with(
                trace_id="test-trace-000",
                name="user_feedback",
                value=0,
                comment=None,
            )

    @pytest.mark.asyncio
    async def test_feedback_api_invalid_score_rejected(self, api_client: AsyncClient) -> None:
        """POST /api/v1/feedback rejects scores not in {0, 1} with HTTP 422."""
        invalid_payloads = [
            {"trace_id": "tr-1", "score": 2},
            {"trace_id": "tr-2", "score": -1},
            {"trace_id": "tr-3", "score": 5},
        ]
        for p in invalid_payloads:
            response = await api_client.post("/api/v1/feedback", json=p)
            assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_feedback_api_missing_trace_id_rejected(self, api_client: AsyncClient) -> None:
        """POST /api/v1/feedback requires non-empty trace_id."""
        response = await api_client.post("/api/v1/feedback", json={"score": 1})
        assert response.status_code == 422

        response = await api_client.post("/api/v1/feedback", json={"trace_id": "", "score": 1})
        assert response.status_code == 422


class TestChatRoutesLangfuseInstrumentation:
    """Tests verifying chat and chat_stream endpoints pass instrumented config."""

    @pytest.mark.asyncio
    async def test_chat_endpoint_uses_instrumented_config(self, api_client: AsyncClient) -> None:
        """Verify chat_endpoint constructs RunnableConfig with langfuse settings."""
        with patch(
            "src.api.routes_chat.get_langfuse_runnable_config",
            wraps=get_langfuse_runnable_config,
        ) as mock_config_factory:
            payload = {
                "message": "Hello CortexGraph!",
                "thread_id": "test-thread-instrumented",
            }
            response = await api_client.post(
                "/api/v1/chat",
                json=payload,
                headers={"X-Trace-Id": "trace-hdr-123"},
            )
            assert response.status_code == 200
            assert mock_config_factory.call_count == 1
            call_kwargs = mock_config_factory.call_args.kwargs
            assert call_kwargs["thread_id"] == "test-thread-instrumented"
            assert call_kwargs["trace_id"] == "trace-hdr-123"
            assert call_kwargs["phase_tag"] == "chat"
