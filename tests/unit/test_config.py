"""Unit tests for configuration loading and validation."""

from src.config import Settings


def test_settings_default_values() -> None:
    """Verify settings defaults when instantiated."""
    settings = Settings(
        OPENAI_API_KEY="sk-override-test",
        CHROMA_HOST="127.0.0.1",
        CHROMA_PORT=9000,
        LOG_FORMAT="json",
    )
    assert settings.openai_api_key == "sk-override-test"
    assert settings.chroma_host == "127.0.0.1"
    assert settings.chroma_port == 9000
    assert settings.log_format == "json"


def test_settings_log_format_validation() -> None:
    """Verify log_format accepts valid literal values."""
    settings_console = Settings(LOG_FORMAT="console")
    assert settings_console.log_format == "console"

    settings_json = Settings(LOG_FORMAT="json")
    assert settings_json.log_format == "json"
