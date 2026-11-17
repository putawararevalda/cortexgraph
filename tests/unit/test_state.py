"""Unit tests for AgentState schema, Pydantic models, and checkpointer persistence."""

import os
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from src.config import Settings
from src.graph.checkpointer import (
    create_sqlite_checkpointer,
    get_checkpointer,
    verify_thread_id_config,
)
from src.graph.state import (
    AgentState,
    Citation,
    RetrievedChunk,
    SqlArtifact,
    create_initial_state,
)


class TestStateSchemaAndModels:
    """Tests verifying strict Pydantic model validation and AgentState typing."""

    def test_retrieved_chunk_valid(self) -> None:
        chunk = RetrievedChunk(
            chunk_id="chunk-101",
            content="LangGraph supports cyclic architectures.",
            source_path="notes/architecture/langgraph.md",
            heading_path="LangGraph > Overview",
            score=0.92,
            metadata={"source_type": "markdown", "tags": "langgraph,agent"},
        )
        assert chunk.chunk_id == "chunk-101"
        assert chunk.content == "LangGraph supports cyclic architectures."
        assert chunk.source_path == "notes/architecture/langgraph.md"
        assert chunk.heading_path == "LangGraph > Overview"
        assert chunk.score == 0.92
        assert chunk.metadata["source_type"] == "markdown"

    def test_retrieved_chunk_strict_validation(self) -> None:
        """Verify extra unexpected fields are rejected by strict validation."""
        with pytest.raises(ValidationError):
            RetrievedChunk(  # type: ignore[call-arg]
                chunk_id="c1",
                content="test",
                source_path="test.md",
                unexpected_field="disallowed",
            )

    def test_citation_valid(self) -> None:
        citation = Citation(
            source_path="notes/architecture/adr-01.md",
            heading_path="Decisions > StateGraph",
            chunk_id="chunk-202",
            quote_snippet="We select LangGraph for deterministic cycles.",
        )
        assert citation.source_path == "notes/architecture/adr-01.md"
        assert citation.chunk_id == "chunk-202"
        assert citation.quote_snippet == "We select LangGraph for deterministic cycles."

    def test_citation_strict_validation(self) -> None:
        with pytest.raises(ValidationError):
            Citation(  # type: ignore[call-arg]
                source_path="notes/test.md",
                chunk_id="c1",
                invalid_key="not_allowed",
            )

    def test_sql_artifact_valid_all_fields(self) -> None:
        artifact = SqlArtifact(
            query="What are the top 5 customers by sales?",
            generated_sql="SELECT customer_name, SUM(total) FROM sales GROUP BY customer_name LIMIT 5;",
            row_count=5,
            data_preview=[
                {"customer_name": "Tailspin Toys", "total": 125000.0},
                {"customer_name": "Wingtip Toys", "total": 98000.0},
            ],
            execution_time_ms=42.5,
        )
        assert artifact.query == "What are the top 5 customers by sales?"
        assert "SELECT customer_name" in artifact.generated_sql
        assert artifact.row_count == 5
        assert len(artifact.data_preview) == 2
        assert artifact.data_preview[0]["customer_name"] == "Tailspin Toys"
        assert artifact.execution_time_ms == 42.5

    def test_sql_artifact_defaults(self) -> None:
        artifact = SqlArtifact(
            query="SELECT count(*) FROM orders;",
            generated_sql="SELECT count(*) FROM orders;",
            row_count=1,
        )
        assert artifact.query == "SELECT count(*) FROM orders;"
        assert artifact.generated_sql == "SELECT count(*) FROM orders;"
        assert artifact.row_count == 1
        assert artifact.data_preview == []
        assert artifact.execution_time_ms == 0.0

    def test_sql_artifact_strict_validation_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            SqlArtifact(  # type: ignore[call-arg]
                query="SELECT 1;",
                generated_sql="SELECT 1;",
                row_count=1,
                disallowed_field="not_permitted",
            )

    def test_sql_artifact_strict_validation_missing_fields(self) -> None:
        with pytest.raises(ValidationError):
            SqlArtifact(  # type: ignore[call-arg]
                query="SELECT 1;",
            )
        with pytest.raises(ValidationError):
            SqlArtifact(  # type: ignore[call-arg]
                generated_sql="SELECT 1;",
                row_count=1,
            )
        with pytest.raises(ValidationError):
            SqlArtifact(  # type: ignore[call-arg]
                query="SELECT 1;",
                generated_sql="SELECT 1;",
            )

    def test_sql_artifact_serialization_roundtrip(self) -> None:
        artifact = SqlArtifact(
            query="Show item count",
            generated_sql="SELECT COUNT(*) AS total FROM warehouse.items;",
            row_count=1,
            data_preview=[{"total": 5230}],
            execution_time_ms=18.4,
        )
        dumped_dict = artifact.model_dump()
        assert dumped_dict["query"] == "Show item count"
        assert dumped_dict["row_count"] == 1
        assert dumped_dict["execution_time_ms"] == 18.4

        json_str = artifact.model_dump_json()
        restored = SqlArtifact.model_validate_json(json_str)
        assert restored == artifact
        assert restored.data_preview == [{"total": 5230}]

    def test_create_initial_state_defaults(self) -> None:
        state = create_initial_state(original_query="What is CRAG?")
        assert state["original_query"] == "What is CRAG?"
        assert state["summary"] == ""
        assert state["intent"] == "chitchat"
        assert state["messages"] == []
        assert state["rewritten_query"] == ""
        assert state["metadata_filters"] == {}
        assert state["retrieved_docs"] == []
        assert state["relevance_status"] == "insufficient"
        assert state["rewrite_count"] == 0
        assert state["generation_count"] == 0
        assert state["citations"] == []
        assert state["is_grounded"] is True
        assert state["hallucination_critique"] is None
        assert state["sql_artifact"] is None

    def test_create_initial_state_with_sql_artifact_and_intent(self) -> None:
        artifact = SqlArtifact(
            query="Show top selling stock items",
            generated_sql="SELECT stock_item_name, quantity FROM warehouse.stock_items ORDER BY quantity DESC LIMIT 3;",
            row_count=3,
            data_preview=[{"stock_item_name": "USB Flash Drive", "quantity": 500}],
            execution_time_ms=25.0,
        )
        state = create_initial_state(
            original_query="Show top selling stock items",
            intent="structured_data_query",
            sql_artifact=artifact,
        )
        assert state["original_query"] == "Show top selling stock items"
        assert state["intent"] == "structured_data_query"
        assert state["sql_artifact"] is not None
        assert state["sql_artifact"].row_count == 3
        assert state["sql_artifact"].data_preview[0]["stock_item_name"] == "USB Flash Drive"


class TestCheckpointerPersistence:
    """Tests verifying checkpointer lifecycle, thread isolation, and reload persistence."""

    def test_verify_thread_id_config(self) -> None:
        with pytest.raises(ValueError, match="Graph invocation requires 'config'"):
            verify_thread_id_config(None)

        with pytest.raises(ValueError, match="Missing 'configurable' dictionary"):
            verify_thread_id_config({})

        with pytest.raises(ValueError, match="requires a non-empty 'thread_id'"):
            verify_thread_id_config({"configurable": {}})

        with pytest.raises(ValueError, match="requires a non-empty 'thread_id'"):
            verify_thread_id_config({"configurable": {"thread_id": "  "}})

        thread_id = verify_thread_id_config({"configurable": {"thread_id": "thread-123"}})
        assert thread_id == "thread-123"

    @pytest.mark.asyncio
    async def test_sqlite_in_memory_persistence(self) -> None:
        def simple_responder(_state: AgentState) -> dict[str, object]:
            return {
                "messages": [AIMessage(content="Acknowledged.")],
                "intent": "chitchat",
            }

        builder = StateGraph(AgentState)
        builder.add_node("responder", simple_responder)
        builder.add_edge(START, "responder")
        builder.add_edge("responder", END)

        async with create_sqlite_checkpointer(":memory:") as checkpointer:
            graph = builder.compile(checkpointer=checkpointer)

            config_1 = {"configurable": {"thread_id": "session-1"}}
            config_2 = {"configurable": {"thread_id": "session-2"}}

            # Session 1 interaction
            await graph.ainvoke(
                {"messages": [HumanMessage(content="Hello from session 1", id="m1")]},
                config=config_1,
            )

            # Session 2 interaction
            await graph.ainvoke(
                {"messages": [HumanMessage(content="Hello from session 2", id="m2")]},
                config=config_2,
            )

            # Check thread isolation
            state_1 = await graph.aget_state(config_1)
            state_2 = await graph.aget_state(config_2)

            assert len(state_1.values["messages"]) == 2
            assert state_1.values["messages"][0].content == "Hello from session 1"

            assert len(state_2.values["messages"]) == 2
            assert state_2.values["messages"][0].content == "Hello from session 2"

    @pytest.mark.asyncio
    async def test_sqlite_disk_persistence_across_process_restart(self, tmp_path: Path) -> None:
        """FR-01: Simulates FastAPI server shutdown and restart with persistent SQLite db."""
        db_path = str(tmp_path / "test_cortex_persistence.db")

        def responder_node(state: AgentState) -> dict[str, object]:
            msg_count = len(state.get("messages", []))
            return {
                "messages": [AIMessage(content=f"Response turn {msg_count}")],
                "summary": f"Summary after turn {msg_count}",
            }

        # --- RUN 1 (Initial server lifecycle) ---
        async with create_sqlite_checkpointer(db_path) as checkpointer_1:
            builder_1 = StateGraph(AgentState)
            builder_1.add_node("responder", responder_node)
            builder_1.add_edge(START, "responder")
            builder_1.add_edge("responder", END)
            graph_1 = builder_1.compile(checkpointer=checkpointer_1)

            thread_config = {"configurable": {"thread_id": "persistent-thread-42"}}
            await graph_1.ainvoke(
                {"messages": [HumanMessage(content="First message before restart", id="msg-1")]},
                config=thread_config,
            )

            state_before = await graph_1.aget_state(thread_config)
            assert len(state_before.values["messages"]) == 2
            assert "Summary after turn 1" in state_before.values["summary"]

        # --- SIMULATE PROCESS TERMINATION & RESTART ---
        # Connection checkpointer_1 is fully closed.
        assert os.path.exists(db_path)

        # --- RUN 2 (New server process lifecycle loading the existing DB) ---
        async with create_sqlite_checkpointer(db_path) as checkpointer_2:
            builder_2 = StateGraph(AgentState)
            builder_2.add_node("responder", responder_node)
            builder_2.add_edge(START, "responder")
            builder_2.add_edge("responder", END)
            graph_2 = builder_2.compile(checkpointer=checkpointer_2)

            # Retrieve state before sending new message
            restored_state = await graph_2.aget_state(thread_config)
            assert len(restored_state.values["messages"]) == 2
            assert "Summary after turn 1" in restored_state.values["summary"]

            # Send follow-up message into the same thread_id
            await graph_2.ainvoke(
                {"messages": [HumanMessage(content="Second message after restart", id="msg-2")]},
                config=thread_config,
            )

            updated_state = await graph_2.aget_state(thread_config)
            # Total messages should now be 4 (2 from run 1 + 2 from run 2)
            assert len(updated_state.values["messages"]) == 4
            assert updated_state.values["messages"][0].content == "First message before restart"
            assert updated_state.values["messages"][2].content == "Second message after restart"

    @pytest.mark.asyncio
    async def test_sqlite_persistence_with_sql_artifact_across_restart(self, tmp_path: Path) -> None:
        """Verify SqlArtifact and structured_data_query intent persist across restarts."""
        db_path = str(tmp_path / "test_sql_artifact_persistence.db")

        def sql_executor_node(_state: AgentState) -> dict[str, object]:
            artifact = SqlArtifact(
                query="What was the Q3 revenue?",
                generated_sql="SELECT SUM(extended_price) FROM sales.invoice_lines;",
                row_count=1,
                data_preview=[{"SUM(extended_price)": 432100.5}],
                execution_time_ms=35.2,
            )
            return {
                "messages": [AIMessage(content="Total Q3 revenue was $432,100.50.")],
                "intent": "structured_data_query",
                "sql_artifact": artifact,
            }

        thread_config = {"configurable": {"thread_id": "sql-thread-99"}}

        # --- RUN 1: Execute graph and persist SqlArtifact ---
        async with create_sqlite_checkpointer(db_path) as checkpointer_1:
            builder_1 = StateGraph(AgentState)
            builder_1.add_node("sql_executor", sql_executor_node)
            builder_1.add_edge(START, "sql_executor")
            builder_1.add_edge("sql_executor", END)
            graph_1 = builder_1.compile(checkpointer=checkpointer_1)

            await graph_1.ainvoke(
                {
                    "messages": [HumanMessage(content="What was the Q3 revenue?", id="msg-q3")],
                    "intent": "structured_data_query",
                },
                config=thread_config,
            )

            state_1 = await graph_1.aget_state(thread_config)
            assert state_1.values["intent"] == "structured_data_query"
            raw_artifact = state_1.values["sql_artifact"]
            if isinstance(raw_artifact, dict):
                loaded_artifact = SqlArtifact.model_validate(raw_artifact)
            else:
                loaded_artifact = raw_artifact
            assert loaded_artifact.row_count == 1
            assert loaded_artifact.generated_sql.startswith("SELECT SUM(extended_price)")
            assert loaded_artifact.data_preview == [{"SUM(extended_price)": 432100.5}]

        # --- RUN 2: Reload DB in fresh checkpointer and verify persistence ---
        async with create_sqlite_checkpointer(db_path) as checkpointer_2:
            builder_2 = StateGraph(AgentState)
            builder_2.add_node("sql_executor", sql_executor_node)
            builder_2.add_edge(START, "sql_executor")
            builder_2.add_edge("sql_executor", END)
            graph_2 = builder_2.compile(checkpointer=checkpointer_2)

            restored_state = await graph_2.aget_state(thread_config)
            assert restored_state.values["intent"] == "structured_data_query"
            restored_raw = restored_state.values["sql_artifact"]
            if isinstance(restored_raw, dict):
                restored_artifact = SqlArtifact.model_validate(restored_raw)
            else:
                restored_artifact = restored_raw
            assert restored_artifact.query == "What was the Q3 revenue?"
            assert restored_artifact.row_count == 1
            assert restored_artifact.execution_time_ms == 35.2

    @pytest.mark.asyncio
    async def test_get_checkpointer_factory(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "factory_test.db")
        settings = Settings(checkpointer_type="sqlite", sqlite_db_path=db_path)

        async with get_checkpointer(settings=settings) as checkpointer:
            assert checkpointer is not None
            assert hasattr(checkpointer, "aget_tuple")
