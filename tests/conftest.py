"""Pytest configuration and shared test fixtures."""

import os
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

# Force offline mock credentials before any application modules read environment
os.environ["OPENAI_API_KEY"] = "sk-mock-deterministic-key-00000000"
os.environ["OPENAI_MODEL"] = "gpt-4o-mini"
os.environ["CHROMA_HOST"] = "localhost"
os.environ["CHROMA_PORT"] = "8001"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-mock-public-key"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-mock-secret-key"
os.environ["LOG_FORMAT"] = "console"
os.environ["DEBUG"] = "false"

from src.api.main import app as fastapi_app
from src.config import Settings, get_settings


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Fixture providing cached application settings with test configuration."""
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
async def async_client() -> AsyncIterator[httpx.AsyncClient]:
    """Asynchronous HTTP test client for FastAPI endpoints."""
    transport = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Deterministic mock client for OpenAI API calls."""
    mock_client = MagicMock()

    # Chat completions mock response builder
    async def fake_create(*_args: Any, **_kwargs: Any) -> MagicMock:
        response = MagicMock()
        response.choices = [
            MagicMock(message=MagicMock(content="Mocked deterministic answer from CortexGraph."))
        ]
        return response

    mock_client.chat.completions.create = AsyncMock(side_effect=fake_create)
    return mock_client


@pytest.fixture
def sample_vault_docs() -> list[dict[str, Any]]:
    """Sample mock document notes for offline RAG testing."""
    return [
        {
            "id": "note-1",
            "content": "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
            "metadata": {"title": "LangGraph Overview", "tags": "ai,orchestration"},
        },
        {
            "id": "note-2",
            "content": "ChromaDB is the AI-native open-source embedding database.",
            "metadata": {"title": "ChromaDB Notes", "tags": "vector,database"},
        },
    ]
