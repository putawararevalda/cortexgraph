"""Application configuration management using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for CortexGraph application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application settings
    app_name: str = "CortexGraph"
    app_version: str = "0.1.0"
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    debug: bool = Field(default=False, alias="DEBUG")

    # OpenAI configuration
    openai_api_key: str = Field(default="sk-test-key-for-local-dev", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )

    # ChromaDB configuration
    chroma_host: str = Field(default="localhost", alias="CHROMA_HOST")
    chroma_port: int = Field(default=8001, alias="CHROMA_PORT")
    chroma_collection_name: str = Field(default="second_brain_notes", alias="CHROMA_COLLECTION_NAME")

    # Langfuse Observability
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", alias="LANGFUSE_HOST")

    # Checkpointer Configuration
    checkpointer_type: Literal["sqlite", "postgres"] = Field(
        default="sqlite", alias="CHECKPOINTER_TYPE"
    )
    sqlite_db_path: str = Field(default="./data/cortex_state.db", alias="SQLITE_DB_PATH")
    postgres_uri: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/cortexgraph",
        alias="POSTGRES_URI",
    )

    # Rolling Conversation Memory & Summarization Limits
    summary_max_messages: int = Field(default=12, alias="SUMMARY_MAX_MESSAGES")
    summary_max_tokens: int = Field(default=4000, alias="SUMMARY_MAX_TOKENS")
    summary_preserve_recent: int = Field(default=4, alias="SUMMARY_PRESERVE_RECENT")

    # NVIDIA Auto-Ontology MCP Configuration
    auto_ontology_mcp_url: str = Field(
        default="http://localhost:3003/mcp", alias="AUTO_ONTOLOGY_MCP_URL"
    )

    # Telemetry and logging
    log_format: Literal["console", "json"] = Field(default="console", alias="LOG_FORMAT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retrieve cached application settings instance."""
    return Settings()
