"""FastAPI chat routes for synchronous and SSE streaming interactions."""

import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

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
from src.observability.langfuse_client import (
    get_langfuse_runnable_config,
    record_feedback,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Chat"])

# Human-friendly progress messages for graph execution nodes
NODE_STATUS_MESSAGES: dict[str, str] = {
    "intent_router": "Analyzing query intent...",
    "query_analyzer": "Analyzing query and metadata...",
    "chroma_retriever": "Searching personal knowledge vault...",
    "document_grader": "Evaluating retrieved notes relevance...",
    "query_rewriter": "Refining search query for better precision...",
    "direct_responder": "Formulating direct answer...",
    "rag_synthesizer": "Synthesizing answer from retrieved notes...",
    "generator": "Generating response...",
    "generate": "Generating response...",
    "hallucination_grader": "Checking groundedness against retrieved notes...",
    "answer_grader": "Validating answer quality...",
    "memory_manager": "Updating conversational memory...",
    "summarizer": "Summarizing conversation turns...",
}


def _format_sse(event_name: str, payload: StreamEvent | dict[str, Any]) -> str:
    """Format an SSE frame into standard text/event-stream wire format."""
    if isinstance(payload, StreamEvent):
        data_str = payload.model_dump_json(exclude_none=True)
    else:
        data_str = json.dumps(payload)
    return f"event: {event_name}\ndata: {data_str}\n\n"


def _extract_text_content(content: Any) -> str:
    """Safely convert message content into a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(str(part["text"]))
        return "".join(parts)
    return str(content) if content is not None else ""


def _parse_citation(item: Any) -> Citation:
    """Convert an arbitrary dict or Citation object to an API Citation."""
    if isinstance(item, Citation):
        return item
    if hasattr(item, "model_dump"):
        return Citation(**item.model_dump())
    if isinstance(item, dict):
        return Citation(**item)
    return Citation(
        source=str(item), source_path=str(item), chunk_id="", content="", quote_snippet=""
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Synchronous conversational query",
    description="Invoke LangGraph agent synchronously and return full response with citations.",
)
async def chat_endpoint(
    payload: ChatRequest,
    request: Request,
    response: Response,
    x_request_id: str | None = Header(default=None, alias="X-Request-Id"),
    x_trace_id: str | None = Header(default=None, alias="X-Trace-Id"),
) -> ChatResponse:
    """Handle synchronous chat requests with stateful checkpointer memory."""
    # Propagate or generate correlation IDs
    req_id = str(x_request_id or getattr(request.state, "request_id", uuid.uuid4().hex))
    trace_id = str(x_trace_id or getattr(request.state, "trace_id", uuid.uuid4().hex))
    response.headers["X-Request-Id"] = req_id
    response.headers["X-Trace-Id"] = trace_id

    thread_id = payload.thread_id or uuid.uuid4().hex
    run_id = uuid.uuid4().hex

    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph execution engine is not initialized.",
        )

    input_state = {
        "messages": [HumanMessage(content=payload.message)],
        "original_query": payload.message,
        "metadata_filters": payload.metadata_filters or {},
    }
    user_id = getattr(request.state, "user_id", None)
    langfuse_config = get_langfuse_runnable_config(
        thread_id=thread_id,
        user_id=user_id,
        trace_id=trace_id,
        phase_tag="chat",
    )
    config: dict[str, Any] = dict(langfuse_config)
    config["configurable"] = {"thread_id": thread_id}
    merged_tags: list[str] = list(langfuse_config.get("tags") or [])
    if "api_v1_chat" not in merged_tags:
        merged_tags.append("api_v1_chat")
    config["tags"] = merged_tags
    merged_metadata: dict[str, Any] = dict(langfuse_config.get("metadata") or {})
    merged_metadata.update({"request_id": req_id, "trace_id": trace_id, "run_id": run_id})
    config["metadata"] = merged_metadata

    try:
        result = await graph.ainvoke(input_state, config=config)
    except Exception as exc:
        logger.error("Chat invocation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing chat graph: {exc}",
        ) from exc

    # Extract final assistant response
    response_text = ""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) or getattr(msg, "type", "") == "ai":
            response_text = _extract_text_content(msg.content)
            break

    # Extract source citations
    raw_citations = result.get("citations", [])
    citations: list[Citation] = [_parse_citation(c) for c in raw_citations]

    return ChatResponse(
        thread_id=thread_id,
        response=response_text,
        citations=citations,
        run_id=run_id,
        metadata={
            "intent": result.get("intent", "chitchat"),
            "relevance_status": result.get("relevance_status", ""),
        },
    )


@router.post(
    "/chat/stream",
    summary="Server-Sent Events (SSE) streaming chat",
    description="Stream real-time graph node progress, LLM tokens, citations, and completion events.",
)
async def chat_stream_endpoint(
    payload: ChatRequest,
    request: Request,
    x_request_id: str | None = Header(default=None, alias="X-Request-Id"),
    x_trace_id: str | None = Header(default=None, alias="X-Trace-Id"),
) -> StreamingResponse:
    """Stream SSE events for real-time progress and token delivery."""
    req_id = str(x_request_id or getattr(request.state, "request_id", uuid.uuid4().hex))
    trace_id = str(x_trace_id or getattr(request.state, "trace_id", uuid.uuid4().hex))

    thread_id = payload.thread_id or uuid.uuid4().hex
    run_id = uuid.uuid4().hex

    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph execution engine is not initialized.",
        )

    input_state = {
        "messages": [HumanMessage(content=payload.message)],
        "original_query": payload.message,
        "metadata_filters": payload.metadata_filters or {},
    }
    user_id = getattr(request.state, "user_id", None)
    langfuse_config = get_langfuse_runnable_config(
        thread_id=thread_id,
        user_id=user_id,
        trace_id=trace_id,
        phase_tag="chat_stream",
    )
    config: dict[str, Any] = dict(langfuse_config)
    config["configurable"] = {"thread_id": thread_id}
    merged_tags: list[str] = list(langfuse_config.get("tags") or [])
    if "api_v1_chat_stream" not in merged_tags:
        merged_tags.append("api_v1_chat_stream")
    config["tags"] = merged_tags
    merged_metadata: dict[str, Any] = dict(langfuse_config.get("metadata") or {})
    merged_metadata.update({"request_id": req_id, "trace_id": trace_id, "run_id": run_id})
    config["metadata"] = merged_metadata

    async def sse_event_generator() -> AsyncIterator[str]:
        tokens_streamed = 0
        citations_emitted = False
        captured_citations: list[Citation] = []

        try:
            async for event in graph.astream_events(input_state, config=config, version="v2"):
                event_type = event.get("event")

                # 1. Node status updates
                if event_type == "on_chain_start":
                    node_name = event.get("metadata", {}).get("langgraph_node")
                    if node_name:
                        msg = NODE_STATUS_MESSAGES.get(
                            node_name,
                            f"Executing {node_name.replace('_', ' ').title()}...",
                        )
                        yield _format_sse(
                            "node_status",
                            StreamEvent(
                                type="node_status",
                                node=node_name,
                                message=msg,
                                thread_id=thread_id,
                                run_id=run_id,
                            ),
                        )

                # 2. LLM streaming tokens
                elif event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    token_str = ""
                    if hasattr(chunk, "content"):
                        token_str = _extract_text_content(chunk.content)
                    elif isinstance(chunk, str):
                        token_str = chunk

                    if token_str:
                        tokens_streamed += 1
                        yield _format_sse(
                            "token",
                            StreamEvent(
                                type="token",
                                content=token_str,
                                thread_id=thread_id,
                                run_id=run_id,
                            ),
                        )

                # 3. Node completion (citations check or fallback token emission)
                elif event_type == "on_chain_end":
                    node_name = event.get("metadata", {}).get("langgraph_node")
                    output = event.get("data", {}).get("output")

                    if isinstance(output, dict):
                        # Extract citations
                        if "citations" in output and output["citations"]:
                            for c in output["citations"]:
                                captured_citations.append(_parse_citation(c))

                        # If this was a generation/responder node and no tokens were streamed yet,
                        # stream response content as tokens so consumers receive full answer
                        if (
                            node_name
                            in ("direct_responder", "rag_synthesizer", "generator", "generate")
                            and tokens_streamed == 0
                            and "messages" in output
                        ):
                            for msg in output["messages"]:
                                if isinstance(msg, AIMessage) or getattr(msg, "type", "") == "ai":
                                    content_str = _extract_text_content(msg.content)
                                    if content_str:
                                        tokens_streamed += 1
                                        yield _format_sse(
                                            "token",
                                            StreamEvent(
                                                type="token",
                                                content=content_str,
                                                thread_id=thread_id,
                                                run_id=run_id,
                                            ),
                                        )

            # 4. Stream citations event if available
            if captured_citations and not citations_emitted:
                citations_emitted = True
                yield _format_sse(
                    "citations",
                    StreamEvent(
                        type="citations",
                        citations=captured_citations,
                        thread_id=thread_id,
                        run_id=run_id,
                    ),
                )

            # 5. Stream done completion event
            yield _format_sse(
                "done",
                StreamEvent(
                    type="done",
                    thread_id=thread_id,
                    run_id=run_id,
                ),
            )

        except Exception as exc:
            logger.error("Error in SSE event stream: %s", exc, exc_info=True)
            yield _format_sse(
                "error",
                StreamEvent(
                    type="error",
                    message=str(exc),
                    thread_id=thread_id,
                    run_id=run_id,
                ),
            )

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Request-Id": req_id,
            "X-Trace-Id": trace_id,
        },
    )


@router.get(
    "/threads/{thread_id}/history",
    response_model=ThreadHistoryResponse,
    summary="Retrieve thread message history",
    description="Fetch full message history and rolling summary for a conversation thread.",
)
async def get_thread_history(
    thread_id: str,
    request: Request,
    response: Response,
    x_request_id: str | None = Header(default=None, alias="X-Request-Id"),
    x_trace_id: str | None = Header(default=None, alias="X-Trace-Id"),
) -> ThreadHistoryResponse:
    """Retrieve checkpointed conversation history for a given thread."""
    req_id = str(x_request_id or getattr(request.state, "request_id", uuid.uuid4().hex))
    trace_id = str(x_trace_id or getattr(request.state, "trace_id", uuid.uuid4().hex))
    response.headers["X-Request-Id"] = req_id
    response.headers["X-Trace-Id"] = trace_id

    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph execution engine is not initialized.",
        )

    config = {"configurable": {"thread_id": thread_id}}
    try:
        state_snapshot = await graph.aget_state(config)
    except Exception as exc:
        logger.error("Failed to fetch state snapshot for thread %s: %s", thread_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading thread state: {exc}",
        ) from exc

    messages: list[ThreadMessage] = []
    summary: str | None = None

    if state_snapshot and state_snapshot.values:
        raw_messages: list[AnyMessage] = state_snapshot.values.get("messages", [])
        summary = state_snapshot.values.get("summary")

        for msg in raw_messages:
            role = "user"
            if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                role = "user"
            elif isinstance(msg, AIMessage) or getattr(msg, "type", "") == "ai":
                role = "assistant"
            elif isinstance(msg, SystemMessage) or getattr(msg, "type", "") == "system":
                role = "system"

            content = _extract_text_content(getattr(msg, "content", ""))
            created_at = None
            if hasattr(msg, "additional_kwargs") and "created_at" in msg.additional_kwargs:
                created_at = str(msg.additional_kwargs["created_at"])

            messages.append(
                ThreadMessage(
                    role=role,
                    content=content,
                    created_at=created_at,
                    metadata=getattr(msg, "additional_kwargs", {}),
                )
            )

    return ThreadHistoryResponse(
        thread_id=thread_id,
        messages=messages,
        summary=summary,
    )


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Submit user feedback score",
    description="Attach binary score (0 or 1) and optional comment to a specific Langfuse trace.",
)
async def submit_feedback(
    payload: FeedbackRequest,
    request: Request,
    response: Response,
    x_request_id: str | None = Header(default=None, alias="X-Request-Id"),
    x_trace_id: str | None = Header(default=None, alias="X-Trace-Id"),
) -> FeedbackResponse:
    """Record user feedback score (0 or 1) against a specific Langfuse trace."""
    req_id = str(x_request_id or getattr(request.state, "request_id", uuid.uuid4().hex))
    trace_id = str(x_trace_id or getattr(request.state, "trace_id", payload.trace_id))
    response.headers["X-Request-Id"] = req_id
    response.headers["X-Trace-Id"] = trace_id

    if payload.score not in (0, 1):
        raise HTTPException(
            status_code=getattr(
                status, "HTTP_422_UNPROCESSABLE_CONTENT", status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail="Score must be strictly 0 (negative) or 1 (positive).",
        )

    record_feedback(
        trace_id=payload.trace_id,
        score=payload.score,
        comment=payload.comment,
    )

    return FeedbackResponse(
        status="ok",
        trace_id=payload.trace_id,
        score=payload.score,
        comment=payload.comment,
    )

