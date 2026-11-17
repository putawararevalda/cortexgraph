"""Unit tests for repository scaffolding, configuration, and health check."""

import pytest
from httpx import AsyncClient

import src
from src.config import Settings


def test_package_metadata() -> None:
    """Verify package exposes valid semver version string."""
    assert hasattr(src, "__version__")
    assert src.__version__ == "0.1.0"


def test_offline_settings(test_settings: Settings) -> None:
    """Ensure settings are populated correctly in zero-cost offline environment."""
    assert test_settings.openai_api_key.startswith("sk-")
    assert test_settings.app_name == "CortexGraph"
    assert test_settings.api_port == 8000
    assert test_settings.chroma_port == 8001


@pytest.mark.asyncio
async def test_health_endpoint(async_client: AsyncClient) -> None:
    """Verify /health endpoint returns HTTP 200 and valid JSON schema."""
    response = await async_client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "CortexGraph"
    assert data["version"] == "0.1.0"
