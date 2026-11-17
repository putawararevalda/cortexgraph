"""Unit tests for rolling conversation memory manager node and token counter."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph import END, START, StateGraph

from src.graph.checkpointer import create_sqlite_checkpointer
from src.graph.nodes import count_message_tokens, memory_manager
from src.graph.state import AgentState, create_initial_state


class TestCountMessageTokens:
    """Tests verifying tiktoken-based message token calculation."""

    def test_count_empty_messages(self) -> None:
        assert count_message_tokens([]) == 0

    def test_count_standard_messages(self) -> None:
        messages = [
            HumanMessage(content="Hello, can you help me find notes on LangGraph?", id="h1"),
            AIMessage(content="Certainly! I will search your notes for LangGraph.", id="a1"),
        ]
        tokens = count_message_tokens(messages)
        # Overhead: 2 msgs * 3 = 6 tokens, plus content tokens
        assert tokens > 15
        assert tokens < 50


class TestMemoryManagerNode:
    """Tests verifying rolling conversation summarization and RemoveMessage trimming."""

    def test_under_threshold_returns_empty(self) -> None:
        """When message count <= 12 and tokens <= 4000, memory_manager does nothing."""
        messages = [HumanMessage(content=f"Message {i}", id=f"msg-{i}") for i in range(8)]
        state = create_initial_state(messages=messages)
        result = memory_manager(state)
        assert result == {}

    def test_exceeding_12_messages_triggers_summarization(self) -> None:
        """FR-02: Conversations exceeding 12 turns summarize older messages, keeping 4 intact."""
        messages = [
            HumanMessage(content=f"Turn {i} question", id=f"msg-h-{i}")
            if i % 2 == 0
            else AIMessage(content=f"Turn {i} response", id=f"msg-a-{i}")
            for i in range(16)
        ]
        state = create_initial_state(messages=messages)
        result = memory_manager(state)

        assert "summary" in result
        assert "messages" in result
        summary = result["summary"]
        assert len(summary) > 0
        assert "Turn 0" in summary

        remove_ops = result["messages"]
        # Total 16 messages: 16 - 4 = 12 messages to prune
        assert len(remove_ops) == 12
        assert all(isinstance(op, RemoveMessage) for op in remove_ops)

        # Verify pruned IDs match the first 12 messages
        pruned_ids = [op.id for op in remove_ops]
        expected_pruned_ids = [m.id for m in messages[:12]]
        assert pruned_ids == expected_pruned_ids

        # The last 4 messages must NOT be in remove_ops
        kept_ids = [m.id for m in messages[12:]]
        assert not any(kept_id in pruned_ids for kept_id in kept_ids)

    def test_exceeding_token_limit_triggers_summarization(self) -> None:
        """Tokens exceeding 4,000 triggers pruning even if message count <= 12."""
        # 6 long messages (~1,000 tokens each = ~6,000 tokens)
        long_text = "LangGraph state management and persistent checkpointing. " * 80
        messages = [
            HumanMessage(content=f"Long message {i}: {long_text}", id=f"long-{i}") for i in range(6)
        ]
        state = create_initial_state(messages=messages)
        assert len(messages) <= 12
        assert count_message_tokens(messages) > 4000

        result = memory_manager(state)
        assert "summary" in result
        assert "messages" in result
        # 6 messages: 6 - 4 = 2 messages to prune, preserving 4
        assert len(result["messages"]) == 2
        pruned_ids = [op.id for op in result["messages"]]
        assert pruned_ids == ["long-0", "long-1"]

    def test_four_or_fewer_messages_never_pruned(self) -> None:
        """Even if tokens exceed threshold, the latest 4 messages must stay intact."""
        long_text = "Word " * 2000
        messages = [
            HumanMessage(content=f"Huge msg {i}: {long_text}", id=f"huge-{i}") for i in range(4)
        ]
        state = create_initial_state(messages=messages)
        assert count_message_tokens(messages) > 4000
        result = memory_manager(state)
        assert result == {}

    def test_memory_manager_with_custom_llm(self) -> None:
        """Verify LLM is used to produce conversational summary when provided."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Synthesized factual summary from LLM."
        mock_llm.invoke.return_value = mock_response

        messages = [HumanMessage(content=f"Fact {i}", id=f"f-{i}") for i in range(14)]
        state = create_initial_state(messages=messages)
        result = memory_manager(state, llm=mock_llm)

        assert mock_llm.invoke.called
        assert result["summary"] == "Synthesized factual summary from LLM."
        assert len(result["messages"]) == 10

    def test_memory_manager_llm_failure_falls_back_gracefully(self) -> None:
        """Verify fallback condensation when LLM raises an unexpected error."""
        failing_llm = MagicMock()
        failing_llm.invoke.side_effect = RuntimeError("OpenAI rate limit / network error")

        messages = [HumanMessage(content=f"Note entry {i}", id=f"note-{i}") for i in range(14)]
        state = create_initial_state(messages=messages)
        result = memory_manager(state, llm=failing_llm)

        # Fallback condensation succeeds without crashing
        assert "summary" in result
        assert "Note entry 0" in result["summary"]
        assert len(result["messages"]) == 10

    def test_memory_manager_preserves_and_extends_existing_summary(self) -> None:
        """Existing summary is carried forward and updated with newly pruned turns."""
        messages = [HumanMessage(content=f"Message {i}", id=f"msg-{i}") for i in range(14)]
        state = create_initial_state(
            messages=messages,
            summary="Prior session context: User works on LangGraph.",
        )
        result = memory_manager(state)

        assert "Prior session context: User works on LangGraph." in result["summary"]
        assert "Message 0" in result["summary"]

    @pytest.mark.asyncio
    async def test_full_stategraph_rolling_memory_integration(self) -> None:
        """Verify rolling memory management inside an active compiled LangGraph workflow."""
        builder = StateGraph(AgentState)
        builder.add_node("memory_manager", memory_manager)
        builder.add_edge(START, "memory_manager")
        builder.add_edge("memory_manager", END)

        async with create_sqlite_checkpointer(":memory:") as checkpointer:
            graph = builder.compile(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": "rolling-thread-test"}}

            # Send 14 turns into the graph
            all_messages = [
                HumanMessage(content=f"Turn message {i}", id=f"turn-{i}") for i in range(14)
            ]
            final_output = await graph.ainvoke({"messages": all_messages}, config=config)

            # Check output values
            remaining_messages = final_output["messages"]
            # After pruning 10 messages (14 - 4 = 10), exactly 4 messages remain
            assert len(remaining_messages) == 4
            assert remaining_messages[0].id == "turn-10"
            assert remaining_messages[3].id == "turn-13"
            assert "Turn message 0" in final_output["summary"]

            # Verify that inspecting the saved checkpoint state matches
            saved_state = await graph.aget_state(config)
            checkpoint_msgs = saved_state.values["messages"]
            assert len(checkpoint_msgs) == 4
            assert checkpoint_msgs[0].id == "turn-10"
            assert checkpoint_msgs[3].id == "turn-13"
            assert "Turn message 0" in saved_state.values["summary"]
