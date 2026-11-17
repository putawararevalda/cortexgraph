"""Unit tests for OpenAI structured output schemas, citation parser, and prompt registry."""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.graph.state import Citation, RetrievedChunk
from src.rag.prompts import (
    INTENT_ROUTER_PROMPT,
    PROMPT_REGISTRY,
    DocumentGrade,
    HallucinationGrade,
    IntentClassification,
    QueryAnalysis,
    QueryRewrite,
    create_sql_citation,
    format_citation,
    format_prompt,
    format_sql_citation,
    get_prompt,
    get_prompt_definition,
    parse_inline_citations,
)


class TestStructuredOutputSchemas:
    """Tests verifying strict Pydantic structured output models for OpenAI."""

    def test_intent_classification_valid(self) -> None:
        valid_intents = ["chitchat", "knowledge_query", "structured_data_query", "clarify"]
        for intent in valid_intents:
            model = IntentClassification(intent=intent, reasoning="Test reason")  # type: ignore[arg-type]
            assert model.intent == intent
            assert model.reasoning == "Test reason"

    def test_intent_classification_invalid_intent_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IntentClassification(intent="unsupported_intent", reasoning="Reason")  # type: ignore[arg-type]

    def test_query_analysis_valid(self) -> None:
        analysis = QueryAnalysis(
            standalone_query="How does LangGraph handle cyclic graphs?",
            source_type_filter="markdown",
            created_after_ts=1728864000,
            required_tags=["langgraph", "architecture"],
        )
        assert analysis.standalone_query == "How does LangGraph handle cyclic graphs?"
        assert analysis.source_type_filter == "markdown"
        assert analysis.created_after_ts == 1728864000
        assert analysis.required_tags == ["langgraph", "architecture"]

    def test_query_analysis_defaults(self) -> None:
        analysis = QueryAnalysis(standalone_query="Simple question")
        assert analysis.standalone_query == "Simple question"
        assert analysis.source_type_filter is None
        assert analysis.created_after_ts is None
        assert analysis.required_tags == []

    def test_document_grade_valid(self) -> None:
        grade = DocumentGrade(
            chunk_id="chunk-42",
            is_relevant=True,
            reason="Contains description of StateGraph nodes and edges.",
        )
        assert grade.chunk_id == "chunk-42"
        assert grade.is_relevant is True
        assert "StateGraph" in grade.reason

    def test_document_grade_missing_required_fields_fails(self) -> None:
        with pytest.raises(ValidationError):
            DocumentGrade(chunk_id="chunk-1")  # type: ignore[call-arg]

    def test_hallucination_grade_valid(self) -> None:
        grade = HallucinationGrade(
            is_grounded=True,
            addresses_query=True,
            critique=None,
        )
        assert grade.is_grounded is True
        assert grade.addresses_query is True
        assert grade.critique is None

    def test_hallucination_grade_with_critique(self) -> None:
        grade = HallucinationGrade(
            is_grounded=False,
            addresses_query=True,
            critique="The claim that LangGraph requires Redis is hallucinated.",
        )
        assert grade.is_grounded is False
        assert grade.critique is not None
        assert "Redis" in grade.critique

    def test_query_rewrite_valid(self) -> None:
        rewrite = QueryRewrite(
            rewritten_query="LangGraph cyclic graph state management",
            relax_filters=True,
            source_type_filter=None,
            required_tags=[],
            reasoning="Broadening query by removing tag restrictions.",
        )
        assert rewrite.rewritten_query == "LangGraph cyclic graph state management"
        assert rewrite.relax_filters is True
        assert rewrite.source_type_filter is None
        assert rewrite.required_tags == []


class TestCitationFormattingAndParsing:
    """Tests verifying FR-05 inline citation formatting and regex parsing."""

    def test_format_citation_with_heading(self) -> None:
        citation_str = format_citation(
            source_path="notes/architecture/langgraph.md",
            heading_path="Overview > Cyclic Graphs",
        )
        assert citation_str == "[Source: notes/architecture/langgraph.md#Overview > Cyclic Graphs]"

    def test_format_citation_without_heading(self) -> None:
        citation_str = format_citation(source_path="notes/adr/adr-001.md")
        assert citation_str == "[Source: notes/adr/adr-001.md]"

    def test_parse_inline_citations_matches_retrieved_docs(self) -> None:
        chunks = [
            RetrievedChunk(
                chunk_id="chunk-1",
                content="LangGraph supports cyclic architectures.",
                source_path="notes/architecture/langgraph.md",
                heading_path="Overview",
                score=0.95,
            ),
            RetrievedChunk(
                chunk_id="chunk-2",
                content="Checkpointers persist state in SQLite or Postgres.",
                source_path="notes/persistence/sqlite.md",
                heading_path="Database Setup",
                score=0.88,
            ),
        ]

        text = (
            "LangGraph supports cyclic architectures [Source: notes/architecture/langgraph.md#Overview]. "
            "State is persisted in SQLite [Source: notes/persistence/sqlite.md#Database Setup]."
        )

        citations = parse_inline_citations(text, retrieved_docs=chunks)

        assert len(citations) == 2
        assert citations[0].source_path == "notes/architecture/langgraph.md"
        assert citations[0].heading_path == "Overview"
        assert citations[0].chunk_id == "chunk-1"
        assert "cyclic" in citations[0].quote_snippet

        assert citations[1].source_path == "notes/persistence/sqlite.md"
        assert citations[1].heading_path == "Database Setup"
        assert citations[1].chunk_id == "chunk-2"

    def test_parse_inline_citations_without_heading(self) -> None:
        chunks = [
            RetrievedChunk(
                chunk_id="chunk-3",
                content="FastAPI SSE streaming chat endpoint.",
                source_path="notes/api.md",
                heading_path="",
                score=0.91,
            )
        ]

        text = "SSE streaming is supported [Source: notes/api.md]."
        citations = parse_inline_citations(text, retrieved_docs=chunks)

        assert len(citations) == 1
        assert citations[0].source_path == "notes/api.md"
        assert citations[0].heading_path == ""
        assert citations[0].chunk_id == "chunk-3"

    def test_parse_inline_citations_deduplication(self) -> None:
        chunks = [
            RetrievedChunk(
                chunk_id="chunk-1",
                content="Content chunk 1",
                source_path="notes/doc.md",
                heading_path="Section A",
            )
        ]
        # Same citation appears twice
        text = (
            "First claim [Source: notes/doc.md#Section A]. "
            "Second claim [Source: notes/doc.md#Section A]."
        )
        citations = parse_inline_citations(text, retrieved_docs=chunks)
        assert len(citations) == 1

    def test_parse_inline_citations_empty_text(self) -> None:
        assert parse_inline_citations("") == []
        assert parse_inline_citations("Text with no citations at all.") == []


class TestPromptRegistryAndLangfuseCompatibility:
    """Tests verifying prompt registry definitions, versioning, and Langfuse fallback."""

    def test_all_seven_llm_nodes_registered(self) -> None:
        expected_nodes = [
            "intent_router",
            "query_analyzer",
            "document_grader",
            "query_rewriter",
            "rag_synthesizer",
            "hallucination_grader",
            "direct_responder",
        ]
        for node in expected_nodes:
            assert node in PROMPT_REGISTRY
            definition = get_prompt_definition(node)
            assert definition.name == node
            assert definition.version in ("1.0.0", "1.1.0")
            assert len(definition.system_template) > 50

    def test_get_prompt_local_fallback(self) -> None:
        prompt = get_prompt("rag_synthesizer")
        assert "EXCLUSIVELY" in prompt
        assert "Mandatory Inline Citations" in prompt
        assert "[Source: {source_path}#{heading_path}]" in prompt
        assert "Abstention Guardrail" in prompt

    def test_get_prompt_langfuse_success(self) -> None:
        mock_client = MagicMock()
        mock_remote_prompt = MagicMock()
        mock_remote_prompt.prompt = "Remote managed prompt from Langfuse"
        mock_client.get_prompt.return_value = mock_remote_prompt

        prompt = get_prompt("intent_router", langfuse_client=mock_client)
        assert prompt == "Remote managed prompt from Langfuse"
        mock_client.get_prompt.assert_called_once_with(name="intent_router", version=None)

    def test_get_prompt_langfuse_failure_falls_back_to_local(self) -> None:
        mock_client = MagicMock()
        mock_client.get_prompt.side_effect = RuntimeError("Langfuse server unreachable")

        prompt = get_prompt("intent_router", langfuse_client=mock_client)
        assert "intent classifier" in prompt.lower()

    def test_format_prompt_variable_substitution(self) -> None:
        sys_txt, usr_txt = format_prompt(
            "document_grader",
            query="What is AST parsing?",
            chunk_id="chunk-77",
            source_path="notes/parser.md",
            heading_path="AST Section",
            content="AST parser splits Markdown by headers.",
        )
        assert "What is AST parsing?" in usr_txt
        assert "chunk-77" in usr_txt
        assert "notes/parser.md" in usr_txt
        assert "relevance evaluator" in sys_txt


class TestSqlCitationFormattingAndParsing:
    """Tests verifying SQL citation formatting, models, and inline extraction."""

    def test_format_sql_citation_basic_table(self) -> None:
        """Verify format_sql_citation with table_name only."""
        citation_str = format_sql_citation(table_name="orders")
        assert citation_str == "[Source: db://wideworldimporters#orders]"

    def test_format_sql_citation_custom_database(self) -> None:
        """Verify format_sql_citation with custom database and prefix stripping."""
        cit1 = format_sql_citation(table_name="sales.customers", db_name="wideworldimporters")
        assert cit1 == "[Source: db://wideworldimporters#sales.customers]"

        cit2 = format_sql_citation(table_name="sales.customers", db_name="db://wideworldimporters")
        assert cit2 == "[Source: db://wideworldimporters#sales.customers]"

    def test_format_sql_citation_with_sql_code_block(self) -> None:
        """Verify format_sql_citation produces executed SQL code block and inline citation."""
        sql = "SELECT StockItemID, StockItemName FROM warehouse.stock_items WHERE QuantityOnHand < 50"
        citation_str = format_sql_citation(
            table_name="warehouse.stock_items",
            sql_query=sql,
            include_code_block=True,
        )
        assert "```sql\n" + sql + "\n```" in citation_str
        assert "[Source: db://wideworldimporters#warehouse.stock_items]" in citation_str

    def test_format_sql_citation_without_code_block(self) -> None:
        """Verify format_sql_citation without code block when include_code_block=False."""
        citation_str = format_sql_citation(
            table_name="warehouse.stock_items",
            sql_query="SELECT 1",
            include_code_block=False,
        )
        assert citation_str == "[Source: db://wideworldimporters#warehouse.stock_items]"
        assert "```" not in citation_str

    def test_format_sql_citation_as_citation_model(self) -> None:
        """Verify format_sql_citation with as_citation=True returns auditable Citation model."""
        sql = "SELECT COUNT(*) FROM sales.orders"
        cit = format_sql_citation("sales.orders", sql_query=sql, as_citation=True)
        assert isinstance(cit, Citation)
        assert cit.source_path == "db://wideworldimporters"
        assert cit.heading_path == "sales.orders"
        assert cit.chunk_id == "sql_execution"
        assert cit.quote_snippet == sql

    def test_create_sql_citation(self) -> None:
        """Verify create_sql_citation generates Citation object matching CortexGraph specs."""
        sql = "SELECT CustomerID, CustomerName FROM sales.customers ORDER BY CustomerID LIMIT 10;"
        cit = create_sql_citation(
            table_name="sales.customers",
            sql_query=sql,
            db_name="wideworldimporters",
        )
        assert cit.source_path == "db://wideworldimporters"
        assert cit.heading_path == "sales.customers"
        assert cit.chunk_id == "sql_execution"
        assert cit.quote_snippet == sql

    def test_parse_inline_citations_for_sql_sources(self) -> None:
        """Verify parse_inline_citations links db:// sources to sql_execution chunk_id."""
        text = (
            "Total Q3 sales reached $4.2M [Source: db://wideworldimporters#sales.invoices]. "
            "Top customer was Tailspin Toys [Source: db://wideworldimporters#sales.customers]."
        )
        citations = parse_inline_citations(text)
        assert len(citations) == 2
        assert citations[0].source_path == "db://wideworldimporters"
        assert citations[0].heading_path == "sales.invoices"
        assert citations[0].chunk_id == "sql_execution"
        assert citations[1].source_path == "db://wideworldimporters"
        assert citations[1].heading_path == "sales.customers"
        assert citations[1].chunk_id == "sql_execution"

    def test_parse_inline_citations_with_sql_code_block_snippet(self) -> None:
        """Verify parse_inline_citations extracts SQL code block into quote_snippet."""
        text = (
            "```sql\nSELECT COUNT(*) FROM sales.orders WHERE Status = 'Shipped'\n```\n"
            "450 orders have shipped [Source: db://wideworldimporters#sales.orders]."
        )
        citations = parse_inline_citations(text)
        assert len(citations) == 1
        assert citations[0].source_path == "db://wideworldimporters"
        assert citations[0].heading_path == "sales.orders"
        assert citations[0].chunk_id == "sql_execution"
        assert "SELECT COUNT(*)" in citations[0].quote_snippet


class TestIntentRouterPromptAndTuning:
    """Tests verifying intent_router prompt instructions, few-shots, and categorization."""

    def test_intent_router_prompt_alias_and_registry(self) -> None:
        """Verify INTENT_ROUTER_PROMPT is exported and registered."""
        assert INTENT_ROUTER_PROMPT is PROMPT_REGISTRY["intent_router"]
        assert INTENT_ROUTER_PROMPT.name == "intent_router"
        assert INTENT_ROUTER_PROMPT.version == "1.1.0"

    def test_intent_router_prompt_distinguishes_notes_from_databases(self) -> None:
        """Verify system prompt contains guidance for structured SQL database vs unstructured notes."""
        prompt_text = get_prompt("intent_router")
        # Structured data guidance
        assert "structured_data_query" in prompt_text
        assert "wideworldimporters" in prompt_text.lower()
        assert "sales figures" in prompt_text.lower() or "sales" in prompt_text.lower()
        assert "inventory levels" in prompt_text.lower() or "inventory" in prompt_text.lower()
        assert "order histories" in prompt_text.lower() or "orders" in prompt_text.lower()
        assert "database metrics" in prompt_text.lower() or "metrics" in prompt_text.lower()

        # Knowledge notes guidance
        assert "knowledge_query" in prompt_text
        assert "personal notes" in prompt_text.lower()
        assert "obsidian" in prompt_text.lower()
        assert "architecture decisions" in prompt_text.lower() or "adr" in prompt_text.lower()

    def test_intent_router_prompt_sample_database_questions_formatting(self) -> None:
        """Verify prompt formatting for diverse sample database queries."""
        sample_db_queries = [
            "What were our total sales numbers for Q3 across all territories?",
            "Show me the top 5 customers by order volume this year.",
            "How many items of 'USB Powered Desk Fan' do we currently have in stock in inventory?",
            "List the recent orders placed by Tailspin Toys and their shipment status.",
            "What is the average order value and total revenue per customer category in the database?",
            "Which suppliers delivered the most purchase orders last month?",
        ]
        for query in sample_db_queries:
            sys_prompt, user_prompt = format_prompt("intent_router", summary="None", query=query)
            assert query in user_prompt
            assert "structured_data_query" in sys_prompt
            assert "wideworldimporters" in sys_prompt.lower()

    def test_intent_classification_structured_data_query_model(self) -> None:
        """Verify IntentClassification validates structured_data_query with rationale."""
        classification = IntentClassification(
            intent="structured_data_query",
            reasoning="Query asks for aggregate sales numbers from database tables.",
        )
        assert classification.intent == "structured_data_query"
        assert "sales" in classification.reasoning
