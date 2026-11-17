"""Unit tests for FastAPI SSE streaming chat, synchronous chat, and thread history."""

import json
import time
from typing import Any
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import create_app
from src.api.schemas import StreamEvent


def parse_sse_events(raw_sse_text: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse raw SSE text into a list of (event_name, data_dict) tuples."""
    events: list[tuple[str, dict[str, Any]]] = []
    current_event = "message"
    current_data_lines: list[str] = []

    for line in raw_sse_text.splitlines():
        line = line.strip()
        if not line:
            if current_data_lines:
                data_payload = "\n".join(current_data_lines)
                events.append((current_event, json.loads(data_payload)))
                current_event = "message"
                current_data_lines = []
            continue

        if line.startswith("event:"):
            current_event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            current_data_lines.append(line.split(":", 1)[1].strip())

    if current_data_lines:
        data_payload = "\n".join(current_data_lines)
        events.append((current_event, json.loads(data_payload)))

    return events


@pytest.fixture
async def api_client() -> AsyncClient:
    """AsyncClient fixture with the full application lifespan initialized."""
    test_app = create_app()
    transport = ASGITransport(app=test_app)
    async with (
        test_app.router.lifespan_context(test_app),
        AsyncClient(transport=transport, base_url="http://testserver") as client,
    ):
        yield client


@pytest.mark.asyncio
async def test_health_check(api_client: AsyncClient) -> None:
    """Verify GET /health returns HTTP 200 with status OK."""
    response = await api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "CortexGraph"
    assert "version" in data


@pytest.mark.asyncio
async def test_chat_synchronous_chitchat(api_client: AsyncClient) -> None:
    """Verify POST /api/v1/chat synchronous response for conversational chitchat."""
    payload = {"message": "Hello, who are you?", "thread_id": "test-sync-thread-1"}
    response = await api_client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["thread_id"] == "test-sync-thread-1"
    assert "CortexGraph" in data["response"]
    assert data["run_id"] is not None
    assert isinstance(data["citations"], list)


@pytest.mark.asyncio
async def test_chat_synchronous_knowledge_query(api_client: AsyncClient) -> None:
    """Verify POST /api/v1/chat with a knowledge query returns citations."""
    payload = {
        "message": "Explain CortexGraph system architecture in detail",
        "thread_id": "test-sync-thread-2",
    }
    response = await api_client.post("/api/v1/chat", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["thread_id"] == "test-sync-thread-2"
    assert len(data["response"]) > 0
    assert len(data["citations"]) > 0
    citation = data["citations"][0]
    assert "source_path" in citation or "source" in citation
    assert citation["chunk_id"] != ""


@pytest.mark.asyncio
async def test_chat_stream_sse_chunks_and_events(api_client: AsyncClient) -> None:
    """Verify POST /api/v1/chat/stream emits structured SSE events with valid JSON."""
    payload = {"message": "Hi there!", "thread_id": "test-stream-thread-1"}

    response = await api_client.post("/api/v1/chat/stream", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    raw_sse = response.text
    events = parse_sse_events(raw_sse)
    assert len(events) >= 3

    event_types = [ev[0] for ev in events]
    assert "node_status" in event_types
    assert "token" in event_types
    assert "done" in event_types

    # Validate node_status event payload
    status_events = [ev for ev in events if ev[0] == "node_status"]
    for _, payload_data in status_events:
        model = StreamEvent(**payload_data)
        assert model.type == "node_status"
        assert model.node is not None
        assert model.message is not None

    # Validate token event payload
    token_events = [ev for ev in events if ev[0] == "token"]
    for _, payload_data in token_events:
        model = StreamEvent(**payload_data)
        assert model.type == "token"
        assert model.content is not None

    # Validate done event payload
    done_events = [ev for ev in events if ev[0] == "done"]
    assert len(done_events) == 1
    done_model = StreamEvent(**done_events[0][1])
    assert done_model.type == "done"
    assert done_model.thread_id == "test-stream-thread-1"
    assert done_model.run_id is not None


@pytest.mark.asyncio
async def test_chat_stream_knowledge_citations(api_client: AsyncClient) -> None:
    """Verify POST /api/v1/chat/stream emits event: citations on knowledge queries."""
    payload = {
        "message": "What is the architecture design for CortexGraph notes?",
        "thread_id": "test-stream-thread-2",
    }

    response = await api_client.post("/api/v1/chat/stream", json=payload)
    assert response.status_code == 200

    events = parse_sse_events(response.text)
    event_names = [ev[0] for ev in events]
    assert "citations" in event_names

    citation_events = [ev for ev in events if ev[0] == "citations"]
    citations_data = citation_events[0][1]
    assert "citations" in citations_data
    assert len(citations_data["citations"]) > 0


@pytest.mark.asyncio
async def test_chat_stream_error_handling() -> None:
    """Verify POST /api/v1/chat/stream handles unexpected graph failures with event: error."""
    test_app = create_app()

    # Mock a failing graph
    failing_graph = MagicMock()

    async def fake_astream_events(*_args: Any, **_kwargs: Any) -> Any:
        # Simulate node status then crash
        yield {
            "event": "on_chain_start",
            "metadata": {"langgraph_node": "intent_router"},
        }
        raise RuntimeError("Simulated internal graph failure")

    failing_graph.astream_events = fake_astream_events

    transport = ASGITransport(app=test_app)
    async with (
        test_app.router.lifespan_context(test_app),
        AsyncClient(transport=transport, base_url="http://testserver") as client,
    ):
        test_app.state.graph = failing_graph
        response = await client.post("/api/v1/chat/stream", json={"message": "Trigger failure"})
        assert response.status_code == 200

        events = parse_sse_events(response.text)
        event_names = [ev[0] for ev in events]
        assert "error" in event_names

        error_events = [ev for ev in events if ev[0] == "error"]
        assert "Simulated internal graph failure" in error_events[0][1]["message"]


@pytest.mark.asyncio
async def test_header_propagation(api_client: AsyncClient) -> None:
    """Verify X-Request-Id and X-Trace-Id headers are propagated across sync and stream."""
    custom_headers = {
        "X-Request-Id": "custom-req-id-12345",
        "X-Trace-Id": "custom-trace-id-67890",
    }

    # Test sync chat
    resp_sync = await api_client.post(
        "/api/v1/chat",
        json={"message": "Header check"},
        headers=custom_headers,
    )
    assert resp_sync.status_code == 200
    assert resp_sync.headers.get("x-request-id") == "custom-req-id-12345"
    assert resp_sync.headers.get("x-trace-id") == "custom-trace-id-67890"

    # Test stream chat
    resp_stream = await api_client.post(
        "/api/v1/chat/stream",
        json={"message": "Header stream check"},
        headers=custom_headers,
    )
    assert resp_stream.status_code == 200
    assert resp_stream.headers.get("x-request-id") == "custom-req-id-12345"
    assert resp_stream.headers.get("x-trace-id") == "custom-trace-id-67890"


@pytest.mark.asyncio
async def test_thread_history(api_client: AsyncClient) -> None:
    """Verify GET /api/v1/threads/{thread_id}/history retrieves persisted conversation turns."""
    thread_id = "test-history-thread-42"

    # Execute a conversational turn
    post_resp = await api_client.post(
        "/api/v1/chat",
        json={"message": "My secret code is Alpha-7", "thread_id": thread_id},
    )
    assert post_resp.status_code == 200

    # Retrieve history
    hist_resp = await api_client.get(f"/api/v1/threads/{thread_id}/history")
    assert hist_resp.status_code == 200

    history = hist_resp.json()
    assert history["thread_id"] == thread_id
    assert len(history["messages"]) >= 2

    roles = [m["role"] for m in history["messages"]]
    assert "user" in roles
    assert "assistant" in roles
    user_msgs = [m["content"] for m in history["messages"] if m["role"] == "user"]
    assert any("Alpha-7" in m for m in user_msgs)


@pytest.mark.asyncio
async def test_time_to_first_token_chitchat(api_client: AsyncClient) -> None:
    """Verify TTFT (Time-To-First-Token) < 800ms target on direct chitchat flow."""
    payload = {"message": "Hi", "thread_id": "test-ttft-thread"}

    start_time = time.perf_counter()
    response = await api_client.post("/api/v1/chat/stream", json=payload)
    assert response.status_code == 200

    # Parse first token event
    events = parse_sse_events(response.text)
    token_events = [ev for ev in events if ev[0] == "token"]
    assert len(token_events) > 0

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    # In local testing, total round-trip stream duration is typically well under 800ms
    assert elapsed_ms < 800, f"TTFT exceeded 800ms: {elapsed_ms:.2f}ms"
