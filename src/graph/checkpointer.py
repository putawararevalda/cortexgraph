"""Persistent checkpointer factory and lifecycle managers for CortexGraph."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.config import Settings, get_settings

AnyCheckpointer = AsyncSqliteSaver | AsyncPostgresSaver


def verify_thread_id_config(config: dict[str, Any] | None) -> str:
    """Validate that the RunnableConfig contains a valid thread_id in configurable.

    Args:
        config: The configuration dictionary passed to graph invocation.

    Returns:
        The validated thread_id string.

    Raises:
        ValueError: If config or configurable['thread_id'] is missing or empty.
    """
    if config is None:
        raise ValueError("Graph invocation requires 'config' with 'configurable.thread_id'.")
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary.")
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        raise ValueError("Missing 'configurable' dictionary in graph configuration.")
    thread_id = configurable.get("thread_id")
    if not thread_id or not str(thread_id).strip():
        raise ValueError("Graph invocation requires a non-empty 'thread_id' in 'configurable'.")
    return str(thread_id).strip()


@asynccontextmanager
async def create_sqlite_checkpointer(
    db_path: str = "./data/cortex_state.db",
    setup_tables: bool = True,
) -> AsyncIterator[AsyncSqliteSaver]:
    """Async context manager creating and initializing an AsyncSqliteSaver.

    Args:
        db_path: Path to SQLite database file or ':memory:'.
        setup_tables: Whether to run saver.setup() to initialize checkpoint schema.

    Yields:
        Initialized AsyncSqliteSaver instance.
    """
    if db_path != ":memory:" and not db_path.startswith("file:"):
        path = Path(db_path)
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

    async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
        if setup_tables:
            await saver.setup()
        yield saver


@asynccontextmanager
async def create_postgres_checkpointer(
    conn_string: str,
    setup_tables: bool = True,
) -> AsyncIterator[AsyncPostgresSaver]:
    """Async context manager creating and initializing an AsyncPostgresSaver.

    Args:
        conn_string: PostgreSQL connection DSN / URI.
        setup_tables: Whether to run saver.setup() to initialize checkpoint schema.

    Yields:
        Initialized AsyncPostgresSaver instance.
    """
    async with AsyncPostgresSaver.from_conn_string(conn_string) as saver:
        if setup_tables:
            await saver.setup()
        yield saver


@asynccontextmanager
async def get_checkpointer(
    settings: Settings | None = None,
    checkpointer_type: Literal["sqlite", "postgres"] | None = None,
    setup_tables: bool = True,
) -> AsyncIterator[AnyCheckpointer]:
    """Factory creating the appropriate checkpointer based on configuration.

    Supports SQLite for local development and PostgreSQL for Docker/production.

    Args:
        settings: Application settings (defaults to get_settings()).
        checkpointer_type: Explicit override ('sqlite' or 'postgres').
        setup_tables: Whether to automatically initialize database schema.

    Yields:
        An initialized AsyncSqliteSaver or AsyncPostgresSaver.
    """
    app_settings = settings or get_settings()
    backend = checkpointer_type or app_settings.checkpointer_type

    if backend == "postgres":
        async with create_postgres_checkpointer(
            conn_string=app_settings.postgres_uri,
            setup_tables=setup_tables,
        ) as saver:
            yield saver
    else:
        async with create_sqlite_checkpointer(
            db_path=app_settings.sqlite_db_path,
            setup_tables=setup_tables,
        ) as saver:
            yield saver
