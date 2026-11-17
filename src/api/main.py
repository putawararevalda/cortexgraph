"""FastAPI application factory, lifespan management, and global middleware."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api.default_graph import create_default_graph
from src.api.middleware import CorrelationMiddleware
from src.api.routes_chat import router as chat_router
from src.api.routes_ingest import router as ingest_router
from src.config import Settings, get_settings
from src.logging_setup import setup_logging

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    """Schema for health status endpoint response."""

    status: str
    app: str
    version: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan context manager for startup and shutdown routines."""
    settings = get_settings()
    app.state.settings = settings

    # 1. Initialize Checkpointer
    checkpointer: Any = None
    checkpointer_cm: Any = None

    try:
        from src.graph.checkpointer import get_checkpointer

        checkpointer_cm = get_checkpointer(settings=settings)
        checkpointer = await checkpointer_cm.__aenter__()
    except Exception as exc:
        logger.warning(
            "Could not initialize checkpointer from src.graph.checkpointer: %s. Using MemorySaver fallback.",
            exc,
        )
        try:
            from langgraph.checkpoint.memory import MemorySaver

            checkpointer = MemorySaver()
        except Exception as inner_exc:
            logger.error("Failed to initialize fallback MemorySaver: %s", inner_exc)

    app.state.checkpointer = checkpointer

    # 2. Initialize ChromaDB Connection
    chroma_client = None
    try:
        import chromadb

        if settings.chroma_host:
            import socket

            server_ready = False
            try:
                with socket.create_connection(
                    (settings.chroma_host, settings.chroma_port), timeout=0.15
                ):
                    server_ready = True
            except OSError:
                server_ready = False

            if server_ready:
                try:
                    client = chromadb.HttpClient(
                        host=settings.chroma_host,
                        port=settings.chroma_port,
                    )
                    client.heartbeat()
                    chroma_client = client
                    logger.info(
                        "Connected to ChromaDB at %s:%s", settings.chroma_host, settings.chroma_port
                    )
                except Exception as chroma_exc:
                    logger.warning(
                        "Remote ChromaDB at %s:%s connection failed (%s). Falling back to ephemeral in-memory client.",
                        settings.chroma_host,
                        settings.chroma_port,
                        chroma_exc,
                    )
                    chroma_client = chromadb.EphemeralClient()
            else:
                logger.info(
                    "Remote ChromaDB at %s:%s unreachable. Using EphemeralClient for local/testing execution.",
                    settings.chroma_host,
                    settings.chroma_port,
                )
                chroma_client = chromadb.EphemeralClient()
        else:
            chroma_client = chromadb.EphemeralClient()
    except Exception as exc:
        logger.warning("ChromaDB initialization bypassed: %s", exc)

    app.state.chroma_client = chroma_client

    # 3. Initialize / Compile LangGraph execution engine
    graph = None
    try:
        from src.graph.builder import build_graph

        graph = build_graph(checkpointer=checkpointer)
    except (ImportError, AttributeError):
        pass

    if graph is None:
        graph = create_default_graph(checkpointer=checkpointer)

    app.state.graph = graph

    logger.info("FastAPI application lifespan startup completed.")
    yield

    # Shutdown: Teardown checkpointer and persistent resources
    if checkpointer_cm is not None:
        try:
            await checkpointer_cm.__aexit__(None, None, None)
        except Exception as exc:
            logger.warning("Error closing checkpointer context: %s", exc)
    elif checkpointer is not None:
        if hasattr(checkpointer, "aclose"):
            try:
                await checkpointer.aclose()
            except Exception as exc:
                logger.warning("Error closing checkpointer: %s", exc)
        elif hasattr(checkpointer, "close"):
            try:
                checkpointer.close()
            except Exception as exc:
                logger.warning("Error closing checkpointer: %s", exc)

    logger.info("FastAPI application lifespan shutdown completed.")


def create_app(custom_settings: Settings | None = None) -> FastAPI:
    """Factory function to configure and return FastAPI application instance."""
    settings = custom_settings or get_settings()

    # Initialize structured logging pipeline
    setup_logging(log_format=settings.log_format, log_level=settings.log_level)

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Agentic Second Brain & Personal Knowledge Assistant API",
        lifespan=lifespan,
    )

    # Enable CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Middleware: Request correlation contextvars propagation
    application.add_middleware(CorrelationMiddleware)

    # Global Exception Handlers
    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": exc.errors(),
                "message": "Input validation failed.",
            },
        )

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled server exception: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal server error",
                "error": str(exc),
            },
        )

    # Health check endpoint
    @application.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check() -> dict[str, Any]:
        """Healthcheck endpoint for orchestrators and container liveness probes."""
        return {
            "status": "ok",
            "app": settings.app_name,
            "version": settings.app_version,
        }

    # Register chat and streaming API routes
    application.include_router(chat_router)
    application.include_router(ingest_router)

    return application


app = create_app()
