"""Unit tests for LangGraph LLM nodes with mocked OpenAI structured outputs."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.nodes import (
    DocumentGrade,
    HallucinationGrade,
    IntentClassification,
    QueryAnalysis,
    QueryRewrite,
    direct_responder,
    document_grader,
    hallucination_grader,
    intent_router,
    mcp_sql_node,
    query_analyzer,
    query_rewriter,
    rag_synthesizer,
)
from src.graph.state import RetrievedChunk, create_initial_state


class TestIntentRouterNode:
    """Tests verifying intent classification and routing."""

    def test_knowledge_query_intent(self) -> None:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = IntentClassification(
            intent="knowledge_query",
            reasoning="User is asking for architecture documentation on LangGraph.",
        )

        state = create_initial_state(original_query="What is the architecture of CortexGraph?")
        result = intent_router(state, llm=mock_llm)

        assert result["intent"] == "knowledge_query"
        assert result["original_query"] == "What is the architecture of CortexGraph?"
        mock_llm.with_structured_output.assert_called_once_with(IntentClassification)

    def test_chitchat_intent(self) -> None:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = IntentClassification(
            intent="chitchat",
            reasoning="User says hello.",
        )

        state = create_initial_state(messages=[HumanMessage(content="Hello there!")])
        result = intent_router(state, llm=mock_llm)

        assert result["intent"] == "chitchat"
        assert result["original_query"] == "Hello there!"

    def test_clarify_intent(self) -> None:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = IntentClassification(
            intent="clarify",
            reasoning="Query 'it' is completely underspecified.",
        )

        state = create_initial_state(original_query="it")
        result = intent_router(state, llm=mock_llm)

        assert result["intent"] == "clarify"

    def test_structured_data_query_intent(self) -> None:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = IntentClassification(
            intent="structured_data_query",
            reasoning="Query asks for Q3 sales figures from database.",
        )

        state = create_initial_state(original_query="What were our total sales numbers for Q3?")
        result = intent_router(state, llm=mock_llm)

        assert result["intent"] == "structured_data_query"
        assert result["original_query"] == "What were our total sales numbers for Q3?"
        mock_llm.with_structured_output.assert_called_once_with(IntentClassification)

    def test_structured_data_query_fallback_preservation(self) -> None:
        mock_llm = MagicMock()
        mock_llm.with_structured_output.side_effect = RuntimeError("OpenAI API unreachable")

        state = create_initial_state(original_query="Show top 5 customers")
        state["intent"] = "structured_data_query"  # type: ignore[typeddict-item]
        result = intent_router(state, llm=mock_llm)

        assert result["intent"] == "structured_data_query"
        assert result["original_query"] == "Show top 5 customers"


class TestQueryAnalyzerNode:
    """Tests verifying query decontextualization and metadata filter extraction."""

    def test_standalone_query_and_filter_extraction(self) -> None:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = QueryAnalysis(
            standalone_query="CortexGraph StateGraph checkpointer persistence",
            source_type_filter="markdown",
            created_after_ts=1728864000,
            required_tags=["langgraph", "architecture"],
        )

        state = create_initial_state(
            original_query="How does its persistence work in recent markdown notes?",
            summary="User previously asked about CortexGraph StateGraph.",
        )
        result = query_analyzer(state, llm=mock_llm)

        assert result["rewritten_query"] == "CortexGraph StateGraph checkpointer persistence"
        filters = result["metadata_filters"]
        assert filters["source_type"] == "markdown"
        assert filters["created_after"] == 1728864000
        assert filters["tags"] == ["langgraph", "architecture"]
        mock_llm.with_structured_output.assert_called_once_with(QueryAnalysis)


class TestDocumentGraderNode:
    """Tests verifying relevance evaluation and filtering of retrieved chunks."""

    def test_filter_relevant_and_irrelevant_chunks(self) -> None:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        # Return relevant for chunk-1, irrelevant for chunk-2
        mock_structured.invoke.side_effect = [
            DocumentGrade(
                chunk_id="chunk-1",
                is_relevant=True,
                reason="Directly describes checkpointer implementation.",
            ),
            DocumentGrade(
                chunk_id="chunk-2",
                is_relevant=False,
                reason="Discusses Docker deployment, not checkpointer logic.",
            ),
        ]

        chunks = [
            RetrievedChunk(
                chunk_id="chunk-1",
                content="AsyncSqliteSaver persists state checkpoints.",
                source_path="notes/checkpointer.md",
                heading_path="SQLite Saver",
            ),
            RetrievedChunk(
                chunk_id="chunk-2",
                content="Docker compose runs ChromaDB on port 8001.",
                source_path="notes/docker.md",
                heading_path="Containers",
            ),
        ]

        state = create_initial_state(
            original_query="How does SQLite checkpointer work?",
            retrieved_docs=chunks,
        )
        result = document_grader(state, llm=mock_llm)

        assert len(result["retrieved_docs"]) == 1
        assert result["retrieved_docs"][0].chunk_id == "chunk-1"
        assert result["relevance_status"] == "relevant"

    def test_all_chunks_irrelevant_returns_insufficient(self) -> None:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = DocumentGrade(
            chunk_id="chunk-1",
            is_relevant=False,
            reason="Completely unrelated to the query.",
        )

        chunks = [
            RetrievedChunk(
                chunk_id="chunk-1",
                content="Unrelated recipe.",
                source_path="notes/recipes.md",
            )
        ]
        state = create_initial_state(
            original_query="Explain CRAG routing.",
            retrieved_docs=chunks,
        )
        result = document_grader(state, llm=mock_llm)

        assert len(result["retrieved_docs"]) == 0
        assert result["relevance_status"] == "insufficient"

    def test_empty_retrieved_docs_returns_insufficient(self) -> None:
        state = create_initial_state(original_query="Any query", retrieved_docs=[])
        result = document_grader(state)
        assert result["retrieved_docs"] == []
        assert result["relevance_status"] == "insufficient"


class TestQueryRewriterNode:
    """Tests verifying query reformulation and filter relaxation."""

    def test_rewrite_and_filter_relaxation(self) -> None:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = QueryRewrite(
            rewritten_query="Corrective RAG CRAG state transitions",
            relax_filters=True,
            source_type_filter=None,
            required_tags=[],
            reasoning="Initial retrieval with tag 'strict-tag' failed; relaxing tags.",
        )

        state = create_initial_state(
            original_query="What is CRAG?",
            rewritten_query="CRAG definition",
            metadata_filters={"source_type": "markdown", "tags": ["strict-tag"]},
            rewrite_count=0,
        )
        result = query_rewriter(state, llm=mock_llm)

        assert result["rewritten_query"] == "Corrective RAG CRAG state transitions"
        assert result["rewrite_count"] == 1
        # Overly restrictive filters were relaxed
        assert "tags" not in result["metadata_filters"]
        assert "source_type" not in result["metadata_filters"]


class TestRAGSynthesizerNode:
    """Tests verifying FR-05 grounded synthesis, mandatory citations, and abstention."""

    def test_abstention_guardrail_when_zero_relevant_chunks(self) -> None:
        """When retrieved_docs has 0 chunks, must transparently abstain without hallucinating."""
        state = create_initial_state(
            original_query="What is the internal design of the secret engine?",
            retrieved_docs=[],  # 0 chunks after grading / rewrites
            rewrite_count=2,
            generation_count=0,
        )
        result = rag_synthesizer(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        ai_msg = result["messages"][0]
        assert isinstance(ai_msg, AIMessage)
        # Transparent abstention statement
        content_str = str(ai_msg.content).lower()
        assert "could not find any notes" in content_str
        assert "personal vault does not contain" in content_str
        # No citations emitted on abstention
        assert result["citations"] == []
        assert result["generation_count"] == 1

    def test_grounded_synthesis_with_inline_citations(self) -> None:
        """Grounded synthesis extracts inline citations and populates state[citations]."""
        mock_llm = MagicMock()
        llm_response = (
            "LangGraph enables cyclic graphs with conditional routing "
            "[Source: notes/architecture/langgraph.md#StateGraph]. "
            "Checkpointer persistence uses SQLite [Source: notes/persistence/sqlite.md#Setup]."
        )
        mock_llm.invoke.return_value = AIMessage(content=llm_response)

        chunks = [
            RetrievedChunk(
                chunk_id="chunk-101",
                content="LangGraph enables cyclic graphs with conditional routing.",
                source_path="notes/architecture/langgraph.md",
                heading_path="StateGraph",
            ),
            RetrievedChunk(
                chunk_id="chunk-102",
                content="Checkpointer persistence uses SQLite.",
                source_path="notes/persistence/sqlite.md",
                heading_path="Setup",
            ),
        ]

        state = create_initial_state(
            original_query="How does LangGraph persist cyclic states?",
            retrieved_docs=chunks,
            generation_count=0,
        )
        result = rag_synthesizer(state, llm=mock_llm)

        assert len(result["messages"]) == 1
        assert result["messages"][0].content == llm_response
        assert result["generation_count"] == 1

        citations = result["citations"]
        assert len(citations) == 2
        assert citations[0].source_path == "notes/architecture/langgraph.md"
        assert citations[0].heading_path == "StateGraph"
        assert citations[0].chunk_id == "chunk-101"

        assert citations[1].source_path == "notes/persistence/sqlite.md"
        assert citations[1].heading_path == "Setup"
        assert citations[1].chunk_id == "chunk-102"


class TestHallucinationGraderNode:
    """Tests verifying hallucination detection against verified notes."""

    def test_grounded_answer(self) -> None:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = HallucinationGrade(
            is_grounded=True,
            addresses_query=True,
            critique=None,
        )

        chunks = [
            RetrievedChunk(
                chunk_id="c1",
                content="CortexGraph uses ChromaDB for vector retrieval.",
                source_path="notes/chroma.md",
            )
        ]
        state = create_initial_state(
            original_query="What vector store is used?",
            retrieved_docs=chunks,
            messages=[AIMessage(content="CortexGraph uses ChromaDB [Source: notes/chroma.md].")],
        )
        result = hallucination_grader(state, llm=mock_llm)

        assert result["is_grounded"] is True
        assert result["hallucination_grade"] == "grounded"
        assert result["hallucination_critique"] is None

    def test_hallucinated_answer(self) -> None:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = HallucinationGrade(
            is_grounded=False,
            addresses_query=True,
            critique="The answer mentions Pinecone and Redis, which are not present in the notes.",
        )

        chunks = [
            RetrievedChunk(
                chunk_id="c1",
                content="CortexGraph uses ChromaDB.",
                source_path="notes/chroma.md",
            )
        ]
        state = create_initial_state(
            original_query="What vector store is used?",
            retrieved_docs=chunks,
            messages=[AIMessage(content="CortexGraph uses Pinecone and Redis.")],
        )
        result = hallucination_grader(state, llm=mock_llm)

        assert result["is_grounded"] is False
        assert result["hallucination_grade"] == "ungrounded"
        assert "Pinecone" in result["hallucination_critique"]

    def test_abstention_auto_grounded(self) -> None:
        """When 0 retrieved docs exist (abstention), response is treated as grounded."""
        state = create_initial_state(
            original_query="Out of domain question",
            retrieved_docs=[],
            messages=[AIMessage(content="I could not find any notes regarding this topic.")],
        )
        result = hallucination_grader(state)
        assert result["is_grounded"] is True
        assert result["hallucination_grade"] == "grounded"


class TestDirectResponderNode:
    """Tests verifying direct conversational responses."""

    def test_direct_chitchat_response(self) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(
            content="Hello! I am CortexGraph, your personal knowledge assistant."
        )

        state = create_initial_state(
            intent="chitchat",
            original_query="Hi!",
        )
        result = direct_responder(state, llm=mock_llm)

        assert len(result["messages"]) == 1
        assert "CortexGraph" in result["messages"][0].content

    def test_direct_clarify_response(self) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(
            content="Could you please specify which architecture component you are asking about?"
        )

        state = create_initial_state(
            intent="clarify",
            original_query="architecture",
        )
        result = direct_responder(state, llm=mock_llm)

        assert len(result["messages"]) == 1
        assert "Could you please specify" in result["messages"][0].content


class TestMcpSqlNode:
    """Tests verifying mcp_sql_node execution, telemetry, citations, and graceful fallback."""

    @pytest.mark.asyncio
    async def test_mcp_sql_node_successful_execution(self) -> None:
        mock_client = AsyncMock()
        mock_client.check_answerable.return_value = True
        mock_client.ask_question.return_value = {
            "answer": "Wide World Importers has 663 customers.",
            "sql": 'SELECT COUNT("CustomerID") AS customer_count FROM "SALES"."CUSTOMERS"',
            "rows": [{"customer_count": 663}],
            "row_count": 1,
            "latency_ms": 45.2,
        }

        state = create_initial_state(
            intent="structured_data_query",
            original_query="How many total customers do we have?",
        )

        result = await mcp_sql_node(state, client=mock_client)

        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert "Wide World Importers has 663 customers." in msg.content
        assert 'SELECT COUNT("CustomerID") AS customer_count FROM "SALES"."CUSTOMERS"' in msg.content
        assert "[Source: db://wideworldimporters#CUSTOMERS]" in msg.content

        # SqlArtifact validation
        artifact = result["sql_artifact"]
        assert artifact is not None
        assert artifact.query == "How many total customers do we have?"
        assert artifact.generated_sql == 'SELECT COUNT("CustomerID") AS customer_count FROM "SALES"."CUSTOMERS"'
        assert artifact.row_count == 1
        assert artifact.data_preview == [{"customer_count": 663}]
        assert artifact.execution_time_ms == 45.2

        # Citations validation
        assert len(result["citations"]) == 1
        citation = result["citations"][0]
        assert citation.source_path == "db://wideworldimporters"
        assert citation.heading_path == "CUSTOMERS"
        assert citation.chunk_id == "sql_execution"

        mock_client.check_answerable.assert_called_once_with("How many total customers do we have?")
        mock_client.ask_question.assert_called_once_with("How many total customers do we have?")

    @pytest.mark.asyncio
    async def test_mcp_sql_node_unanswerable_fallback(self) -> None:
        mock_client = AsyncMock()
        mock_client.check_answerable.return_value = False

        state = create_initial_state(
            intent="structured_data_query",
            original_query="What is the weather in Tokyo tomorrow?",
        )

        result = await mcp_sql_node(state, client=mock_client)

        assert len(result["messages"]) == 1
        content = result["messages"][0].content
        assert "cannot answer" in content
        assert "semantic layer" in content
        assert result["sql_artifact"] is None
        assert result["citations"] == []
        mock_client.ask_question.assert_not_called()

    @pytest.mark.asyncio
    async def test_mcp_sql_node_error_fallback(self) -> None:
        mock_client = AsyncMock()
        mock_client.check_answerable.return_value = True
        mock_client.ask_question.side_effect = RuntimeError("MCP connection timed out")

        state = create_initial_state(
            intent="structured_data_query",
            original_query="Show top 5 sales orders",
        )

        result = await mcp_sql_node(state, client=mock_client)

        assert len(result["messages"]) == 1
        assert "Unable to retrieve structured data" in result["messages"][0].content
        assert result["sql_artifact"] is None
        assert result["citations"] == []

    @pytest.mark.asyncio
    async def test_mcp_sql_node_sync_mock_compatibility(self) -> None:
        """Verifies mcp_sql_node works cleanly with standard MagicMock."""
        mock_client = MagicMock()
        mock_client.check_answerable.return_value = True
        mock_client.ask_question.return_value = {
            "answer": "Active orders count: 12",
            "sql": "SELECT * FROM orders WHERE status = 'active'",
            "rows": [{"id": 1}],
            "row_count": 1,
            "latency_ms": 10.0,
        }

        state = create_initial_state(
            intent="structured_data_query",
            original_query="Count active orders",
        )

        result = await mcp_sql_node(state, client=mock_client)
        assert result["sql_artifact"] is not None
        assert result["sql_artifact"].row_count == 1
        assert len(result["citations"]) == 1
