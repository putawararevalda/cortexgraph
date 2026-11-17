"""Unit tests for Phase 3: Structlog JSON telemetry, secret scrubbing, ASGI contextvars, and Tenacity retries."""

import asyncio
import io
import json
from typing import Any

import httpx
import openai
import pytest
import structlog
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import HumanMessage

from src.api.middleware import (
    CorrelationMiddleware,
    bind_correlation_context,
    get_correlation_context,
)
from src.graph.builder import build_graph
from src.graph.state import create_initial_state
from src.logging_setup import (
    SECRET_PATTERN,
    is_sensitive_key,
    log_node_execution,
    redact_secrets_processor,
    setup_logging,
)
from src.rag.resiliency import (
    retry_openai_call,
)


class TestSecretRedaction:
    """Tests verifying zero-leak secret redaction across log processors and structures."""

    def test_secret_pattern_matching(self) -> None:
        """Verify regex correctly matches dummy and realistic API key formats."""
        assert SECRET_PATTERN.search("sk-1234567890abcdef") is not None
        assert SECRET_PATTERN.search("pk-lf-mock-public-key-99") is not None
        assert SECRET_PATTERN.search("sk-lf-mock-secret-key-42") is not None
        assert SECRET_PATTERN.search("sk-proj-abc123XYZ") is not None
        assert SECRET_PATTERN.search("harmless-string-without-keys") is None

    def test_is_sensitive_key(self) -> None:
        """Verify sensitive key name detection."""
        assert is_sensitive_key("api_key") is True
        assert is_sensitive_key("API_KEY") is True
        assert is_sensitive_key("authorization") is True
        assert is_sensitive_key("openai_api_key") is True
        assert is_sensitive_key("langfuse_secret_key") is True
        assert is_sensitive_key("sk-dummy-key-header") is True
        assert is_sensitive_key("pk-lf-public") is True
        assert is_sensitive_key("query") is False
        assert is_sensitive_key("rewrite_count") is False

    def test_redact_dummy_keys_and_values(self) -> None:
        """Verify passing dummy sk-... keys into log events masks them as [REDACTED]."""
        raw_event = {
            "event": "client_authenticated",
            "api_key": "raw-key-value",
            "openai_api_key": "sk-dummy-openai-key-12345",
            "langfuse_secret_key": "sk-lf-mock-secret-98765",
            "authorization": "Bearer sk-dummy-bearer-token",
            "sk-dummy-named-key": "secret_content",
            "pk-lf-key": "public_val",
            "regular_field": "unmasked_value",
            "mixed_text": "Failed connecting with sk-leaked-key-in-error",
        }

        redacted = redact_secrets_processor(None, "info", raw_event)

        assert redacted["api_key"] == "[REDACTED]"
        assert redacted["openai_api_key"] == "[REDACTED]"
        assert redacted["langfuse_secret_key"] == "[REDACTED]"
        assert redacted["authorization"] == "[REDACTED]"
        assert redacted["sk-dummy-named-key"] == "[REDACTED]"
        assert redacted["pk-lf-key"] == "[REDACTED]"
        assert redacted["regular_field"] == "unmasked_value"
        assert "sk-leaked-key-in-error" not in redacted["mixed_text"]
        assert "[REDACTED]" in redacted["mixed_text"]

    def test_recursive_nested_redaction(self) -> None:
        """Verify nested dictionaries, lists, and tuples are scrubbed recursively."""
        nested_event = {
            "event": "batch_process",
            "metadata": {
                "user": {"token": "sk-user-token-000"},
                "headers": {"Authorization": "Bearer sk-header-token-111"},
            },
            "keys_list": ["sk-item-1", "normal-item", "pk-lf-item-2"],
            "keys_tuple": ("sk-tuple-1", "plain-tuple"),
        }

        redacted = redact_secrets_processor(None, "info", nested_event)

        assert redacted["metadata"]["user"]["token"] == "[REDACTED]"
        assert redacted["metadata"]["headers"]["Authorization"] == "[REDACTED]"
        assert redacted["keys_list"][0] == "[REDACTED]"
        assert redacted["keys_list"][1] == "normal-item"
        assert redacted["keys_list"][2] == "[REDACTED]"
        assert redacted["keys_tuple"][0] == "[REDACTED]"
        assert redacted["keys_tuple"][1] == "plain-tuple"


class TestStructuredLoggingPipeline:
    """Tests verifying JSON / console formatting and contextvars binding."""

    def test_json_log_formatting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify LOG_FORMAT=json emits valid single-line JSON with correlation keys."""
        buffer = io.StringIO()
        monkeypatch.setattr("sys.stdout", buffer)

        setup_logging(log_format="json", log_level="INFO")
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id="test-req-101",
            trace_id="test-trace-202",
            thread_id="test-thread-303",
        )

        logger = structlog.get_logger("test.json")
        logger.info("structured_telemetry_event", metric_score=0.95)

        output = buffer.getvalue().strip()
        assert output != ""
        parsed = json.loads(output.splitlines()[-1])

        assert parsed["event"] == "structured_telemetry_event"
        assert parsed["request_id"] == "test-req-101"
        assert parsed["trace_id"] == "test-trace-202"
        assert parsed["thread_id"] == "test-thread-303"
        assert parsed["metric_score"] == 0.95
        assert parsed["level"] == "info"

    def test_console_log_formatting(self) -> None:
        """Verify LOG_FORMAT=console configures console renderer without errors."""
        setup_logging(log_format="console", log_level="DEBUG")
        logger = structlog.get_logger("test.console")
        logger.debug("console_dev_event", sample_param=123)


class TestNodeLifecycleInstrumentationDecorator:
    """Tests verifying @log_node_execution decorator lifecycle, timing, and delta tracking."""

    def test_sync_node_execution_telemetry(self) -> None:
        """Verify sync node logs node_started and node_completed with duration_ms and state keys."""
        events_captured: list[dict[str, Any]] = []

        def capture_processor(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
            events_captured.append(dict(event_dict))
            return event_dict

        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                capture_processor,
                structlog.processors.JSONRenderer(),
            ]
        )

        @log_node_execution("test_sync_node")
        def sample_sync_node(_state: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
            return {
                "intent": "knowledge_query",
                "rewrite_count": 1,
                "notes": ["chunk-1"],
            }

        input_state = {"original_query": "hello", "rewrite_count": 0, "thread_id": "thread-test-sync"}
        result = sample_sync_node(input_state)

        assert result["intent"] == "knowledge_query"
        assert len(events_captured) == 2

        start_event = events_captured[0]
        assert start_event["event"] == "node_started"
        assert start_event["node_name"] == "test_sync_node"
        assert start_event["rewrite_count"] == 0

        complete_event = events_captured[1]
        assert complete_event["event"] == "node_completed"
        assert complete_event["node_name"] == "test_sync_node"
        assert complete_event["rewrite_count"] == 1
        assert isinstance(complete_event["duration_ms"], float)
        assert complete_event["duration_ms"] >= 0.0
        assert "intent" in complete_event["updated_state_keys"]
        assert "notes" in complete_event["updated_state_keys"]

    @pytest.mark.asyncio
    async def test_async_node_execution_telemetry(self) -> None:
        """Verify async node logs timing, duration_ms, and updated state keys."""
        events_captured: list[dict[str, Any]] = []

        def capture_processor(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
            events_captured.append(dict(event_dict))
            return event_dict

        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                capture_processor,
                structlog.processors.JSONRenderer(),
            ]
        )

        @log_node_execution("test_async_node")
        async def sample_async_node(_state: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
            await asyncio.sleep(0.01)
            return {"async_result": "done"}

        result = await sample_async_node({"thread_id": "t-async"})
        assert result["async_result"] == "done"
        assert len(events_captured) == 2

        complete_event = events_captured[1]
        assert complete_event["event"] == "node_completed"
        assert complete_event["node_name"] == "test_async_node"
        assert complete_event["duration_ms"] >= 9.0
        assert complete_event["updated_state_keys"] == ["async_result"]

    def test_node_failure_logging(self) -> None:
        """Verify failing node logs node_failed event with duration_ms and error before re-raising."""
        events_captured: list[dict[str, Any]] = []

        def capture_processor(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
            events_captured.append(dict(event_dict))
            return event_dict

        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                capture_processor,
                structlog.processors.JSONRenderer(),
            ]
        )

        @log_node_execution("failing_node")
        def crashing_node(_state: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("Database connection timed out")

        with pytest.raises(RuntimeError, match="Database connection timed out"):
            crashing_node({"thread_id": "t-err"})

        assert len(events_captured) == 2
        fail_event = events_captured[1]
        assert fail_event["event"] == "node_failed"
        assert fail_event["node_name"] == "failing_node"
        assert "Database connection timed out" in fail_event["error"]
        assert isinstance(fail_event["duration_ms"], float)


class TestCorrelationMiddleware:
    """Tests verifying ASGI correlation contextvars middleware."""

    @pytest.mark.asyncio
    async def test_correlation_headers_and_contextvars(self) -> None:
        """Verify middleware extracts headers and query params into structlog contextvars."""
        test_app = FastAPI()
        test_app.add_middleware(CorrelationMiddleware)

        captured_context: dict[str, Any] = {}

        @test_app.get("/test-endpoint")
        async def sample_endpoint(request: Request) -> dict[str, Any]:
            nonlocal captured_context
            captured_context = get_correlation_context()
            return {
                "request_id": request.state.request_id,
                "trace_id": request.state.trace_id,
                "thread_id": request.state.thread_id,
                "user_id": request.state.user_id,
            }

        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get(
                "/test-endpoint?user_id=usr-99",
                headers={
                    "X-Request-Id": "custom-req-777",
                    "X-Trace-Id": "custom-trace-888",
                    "X-Thread-Id": "custom-thread-999",
                },
            )

            assert response.status_code == 200
            assert response.headers["X-Request-Id"] == "custom-req-777"
            assert response.headers["X-Trace-Id"] == "custom-trace-888"
            assert response.headers["X-Thread-Id"] == "custom-thread-999"
            assert response.headers["X-User-Id"] == "usr-99"

            # Check contextvars bound during request execution
            assert captured_context["request_id"] == "custom-req-777"
            assert captured_context["trace_id"] == "custom-trace-888"
            assert captured_context["thread_id"] == "custom-thread-999"
            assert captured_context["user_id"] == "usr-99"

    @pytest.mark.asyncio
    async def test_auto_generated_correlation_ids(self) -> None:
        """Verify middleware generates uuid identifiers when headers are omitted."""
        test_app = FastAPI()
        test_app.add_middleware(CorrelationMiddleware)

        @test_app.get("/probe")
        async def probe_endpoint() -> dict[str, str]:
            return {"status": "ok"}

        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/probe")
            assert response.status_code == 200
            assert "X-Request-Id" in response.headers
            assert "X-Trace-Id" in response.headers
            assert len(response.headers["X-Request-Id"]) >= 16
            assert len(response.headers["X-Trace-Id"]) >= 16


class TestTenacityOpenAIRetry:
    """Tests verifying Tenacity exponential retry backoff and structured warning logs."""

    def test_simulated_429_triggers_three_attempts_and_structured_logs(self) -> None:
        """Simulated 429 errors trigger exactly 3 retry attempts with structured warning logs."""
        warnings_logged: list[dict[str, Any]] = []

        def capture_processor(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
            if event_dict.get("event") == "openai_retry_attempt":
                warnings_logged.append(dict(event_dict))
            return event_dict

        structlog.configure(
            processors=[
                capture_processor,
                structlog.processors.JSONRenderer(),
            ]
        )

        call_attempts = 0

        # Construct simulated OpenAI HTTP 429 RateLimitError
        mock_req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        mock_resp = httpx.Response(429, request=mock_req)
        rate_limit_err = openai.RateLimitError(
            message="Rate limit exceeded. Please retry after backoff.",
            response=mock_resp,
            body=None,
        )

        @retry_openai_call(max_attempts=3, min_wait=0.001, max_wait=0.01)
        def failing_openai_call() -> str:
            nonlocal call_attempts
            call_attempts += 1
            raise rate_limit_err

        with pytest.raises(openai.RateLimitError) as exc_info:
            failing_openai_call()

        assert exc_info.value == rate_limit_err
        # Exactly 3 total call attempts executed before raising
        assert call_attempts == 3
        # Exactly 2 retry warning logs emitted before sleep intervals
        assert len(warnings_logged) == 2
        for log_entry in warnings_logged:
            assert log_entry["event"] == "openai_retry_attempt"
            assert log_entry["error_type"] == "RateLimitError"
            assert "Rate limit exceeded" in log_entry["error_message"]
            assert log_entry["sleep_seconds"] >= 0.0

    @pytest.mark.asyncio
    async def test_simulated_connection_error_async_retry(self) -> None:
        """Verify APIConnectionError triggers retries in async functions."""
        attempts = 0
        mock_req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        conn_err = openai.APIConnectionError(request=mock_req)

        @retry_openai_call(max_attempts=3, min_wait=0.001, max_wait=0.01)
        async def failing_async_call() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise conn_err
            return "recovered_on_third_try"

        result = await failing_async_call()
        assert result == "recovered_on_third_try"
        assert attempts == 3

    def test_non_retryable_error_does_not_retry(self) -> None:
        """Verify non-retryable errors fail immediately on first attempt."""
        call_count = 0

        @retry_openai_call(max_attempts=3, min_wait=0.001, max_wait=0.01)
        def bad_params_call() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid model argument")

        with pytest.raises(ValueError, match="Invalid model argument"):
            bad_params_call()

        assert call_count == 1


class TestGraphExecutionStructuredTelemetry:
    """Tests verifying every graph execution emits structured JSON logs with correlation keys."""

    @pytest.mark.asyncio
    async def test_graph_node_execution_emits_required_keys(self) -> None:
        """Every graph execution emits structured JSON logs containing request_id, thread_id, trace_id, node_name, and duration_ms."""
        logs_emitted: list[dict[str, Any]] = []

        def capture_processor(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
            logs_emitted.append(dict(event_dict))
            return event_dict

        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                capture_processor,
                structlog.processors.JSONRenderer(),
            ]
        )

        test_thread = "test-graph-telemetry-thread-10"
        test_req = "req-telemetry-abc"
        test_trace = "trace-telemetry-xyz"

        structlog.contextvars.clear_contextvars()
        bind_correlation_context(
            request_id=test_req,
            trace_id=test_trace,
            thread_id=test_thread,
        )

        # Mock direct chitchat flow through intent_router -> direct_responder -> memory_manager
        graph = build_graph()
        initial_state = create_initial_state(
            messages=[HumanMessage(content="Hello CortexGraph!")],
            original_query="Hello CortexGraph!",
        )

        config = {
            "configurable": {"thread_id": test_thread},
            "metadata": {"request_id": test_req, "trace_id": test_trace},
        }

        result = await graph.ainvoke(initial_state, config=config)
        assert result is not None

        # Filter for node_completed events
        completed_events = [ev for ev in logs_emitted if ev.get("event") == "node_completed"]
        assert len(completed_events) >= 1

        for ev in completed_events:
            assert ev["request_id"] == test_req
            assert ev["trace_id"] == test_trace
            assert ev["thread_id"] == test_thread
            assert "node_name" in ev
            assert "duration_ms" in ev
            assert isinstance(ev["duration_ms"], float)
            assert ev["duration_ms"] >= 0.0
