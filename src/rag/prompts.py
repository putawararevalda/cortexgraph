"""Prompt templates, prompt registry, citation parsing, and structured output schemas."""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.graph.state import Citation, RetrievedChunk

logger = logging.getLogger(__name__)

# ============================================================================
# 1. Strict Structured Output Pydantic Schemas
# ============================================================================


class IntentClassification(BaseModel):
    """Structured classification of user intent for query routing."""

    model_config = ConfigDict(extra="ignore")

    intent: Literal["chitchat", "knowledge_query", "structured_data_query", "clarify"] = Field(
        ...,
        description=(
            "The classified intent category: 'knowledge_query' for questions requiring "
            "vault lookup, personal notes, architecture, or documentation; 'structured_data_query' "
            "for numerical, analytical, aggregate, or transactional queries asking about tables, "
            "orders, products, customers, or metrics from Wide World Importers; 'chitchat' for "
            "casual greeting, identity questions, or conversational small talk; 'clarify' "
            "for ambiguous, incomplete, or underspecified queries."
        ),
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of the rationale for this intent classification.",
    )


class QueryAnalysis(BaseModel):
    """Structured query analysis for retrieval and metadata filtering."""

    model_config = ConfigDict(extra="ignore")

    standalone_query: str = Field(
        ...,
        description=(
            "A clear, self-contained search query resolving pronouns, conversational "
            "anaphora, and context into explicit terms suitable for vector retrieval."
        ),
    )
    source_type_filter: str | None = Field(
        default=None,
        description="Optional source type filter (e.g., 'markdown', 'pdf') if explicitly requested.",
    )
    created_after_ts: int | None = Field(
        default=None,
        description="Optional Unix timestamp filter for note creation date if timeframe requested.",
    )
    required_tags: list[str] = Field(
        default_factory=list,
        description="List of specific tags or topic keywords to filter knowledge chunks by.",
    )


class DocumentGrade(BaseModel):
    """Relevance evaluation score and rationale for a retrieved document chunk."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: str = Field(
        ...,
        description="The unique identifier of the retrieved chunk being evaluated.",
    )
    is_relevant: bool = Field(
        ...,
        description=(
            "True if the chunk contains information, facts, or context directly relevant "
            "to answering the query; False if irrelevant, redundant, or tangential."
        ),
    )
    reason: str = Field(
        ...,
        description="Concise rationale for relevance determination.",
    )


class HallucinationGrade(BaseModel):
    """Evaluation of groundedness and answer relevance against retrieved sources."""

    model_config = ConfigDict(extra="ignore")

    is_grounded: bool = Field(
        ...,
        description=(
            "True if every factual claim or assertion in the generated response is strictly "
            "supported by the retrieved document chunks; False if ungrounded or hallucinated."
        ),
    )
    addresses_query: bool = Field(
        ...,
        description="True if the response directly addresses and answers the user's query.",
    )
    critique: str | None = Field(
        default=None,
        description="Constructive critique detailing any hallucinated, unverified, or omitted facts.",
    )


class QueryRewrite(BaseModel):
    """Structured query reformulation and metadata filter relaxation."""

    model_config = ConfigDict(extra="ignore")

    rewritten_query: str = Field(
        ...,
        description=(
            "Alternative search query reformulated with synonyms, broader technical concepts, "
            "or adjusted phrasing to overcome failed previous retrieval."
        ),
    )
    relax_filters: bool = Field(
        default=False,
        description="Whether overly restrictive metadata filters should be relaxed or removed.",
    )
    source_type_filter: str | None = Field(
        default=None,
        description="Relaxed or broadened source type filter (None to remove restriction).",
    )
    required_tags: list[str] = Field(
        default_factory=list,
        description="Relaxed list of required tags (empty list to remove tag filters).",
    )
    reasoning: str = Field(
        default="",
        description="Reasoning explaining the query reformulation and filter adjustments.",
    )


# ============================================================================
# 2. Inline Citation Formatting & Extraction
# ============================================================================

# Strict regex matching [Source: {source_path}#{heading_path}] or [Source: {source_path}]
CITATION_REGEX = re.compile(r"\[Source:\s*([^#\]]+?)(?:#([^\]]*?))?\]")


def format_citation(source_path: str, heading_path: str = "") -> str:
    """Format an inline citation string following the strict specification.

    Format: `[Source: {source_path}#{heading_path}]` or `[Source: {source_path}]` if no heading.
    """
    clean_src = source_path.strip()
    clean_heading = heading_path.strip()
    if clean_heading:
        return f"[Source: {clean_src}#{clean_heading}]"
    return f"[Source: {clean_src}]"


def create_sql_citation(
    table_name: str,
    sql_query: str = "",
    db_name: str = "wideworldimporters",
    chunk_id: str = "sql_execution",
) -> Citation:
    """Create a deterministically auditable Citation model object for SQL executions.

    Args:
        table_name: Relational database table name.
        sql_query: The executed SQL statement.
        db_name: Database name or URI scheme (defaults to 'wideworldimporters').
        chunk_id: Unique identifier for the execution chunk (defaults to 'sql_execution').

    Returns:
        Strictly validated Citation object compatible with CortexGraph state[citations].
    """
    clean_table = table_name.strip()
    clean_db = db_name.strip().removeprefix("db://")
    source_path = f"db://{clean_db}"
    return Citation(
        source_path=source_path,
        heading_path=clean_table,
        chunk_id=chunk_id,
        quote_snippet=sql_query.strip(),
    )


def format_sql_citation(
    table_name: str,
    sql_query: str = "",
    db_name: str = "wideworldimporters",
    include_code_block: bool = True,
    as_citation: bool = False,
) -> str | Citation:
    """Format an inline SQL citation string or structured Citation model object.

    Produces citations matching `[Source: db://{db_name}#{table_name}]`.
    When `sql_query` is provided and `include_code_block=True`, includes an executed
    SQL markdown code block preceding the inline citation.

    Args:
        table_name: Relational database table name (e.g. 'sales.orders' or 'stock_items').
        sql_query: Optional executed SQL query string.
        db_name: Relational database name or URI scheme (defaults to 'wideworldimporters').
        include_code_block: If True and sql_query is non-empty, includes a ```sql markdown block.
        as_citation: If True, returns a structured Citation object instead of a string.

    Returns:
        Markdown string or structured Citation object compatible with CortexGraph state[citations].
    """
    clean_table = table_name.strip()
    clean_db = db_name.strip().removeprefix("db://")
    source_path = f"db://{clean_db}"
    clean_sql = sql_query.strip()

    if as_citation:
        return create_sql_citation(
            table_name=clean_table,
            sql_query=clean_sql,
            db_name=source_path,
        )

    inline_citation = f"[Source: {source_path}#{clean_table}]"
    if clean_sql and include_code_block:
        return f"```sql\n{clean_sql}\n```\n{inline_citation}"
    return inline_citation


def parse_inline_citations(
    text: str,
    retrieved_docs: Sequence[RetrievedChunk] | None = None,
) -> list[Citation]:
    """Parse all inline citations from text and link them to retrieved document chunks or SQL executions.

    Args:
        text: Synthesized markdown response text containing inline citations.
        retrieved_docs: Verified document chunks available during synthesis.

    Returns:
        List of structured Citation objects matching state[citations].
    """
    if not text:
        return []

    citations: list[Citation] = []
    seen: set[tuple[str, str, str]] = set()

    # Pre-index retrieved docs for fast matching
    doc_list = list(retrieved_docs or [])

    # Pre-extract SQL code block if present for quote snippet fallback
    sql_match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL)
    default_sql_snippet = sql_match.group(1).strip() if sql_match else ""

    for match in CITATION_REGEX.finditer(text):
        source_path = match.group(1).strip()
        heading_path = (match.group(2) or "").strip()

        # Find matching chunk among retrieved docs
        matched_chunk: RetrievedChunk | None = None
        for chunk in doc_list:
            chunk_src = chunk.source_path.strip()
            # Check for exact path match or suffix/relative match
            if chunk_src == source_path or chunk_src.endswith(source_path) or source_path.endswith(chunk_src):
                if heading_path:
                    if chunk.heading_path.strip() == heading_path:
                        matched_chunk = chunk
                        break
                    # Keep as candidate if no better heading match found
                    if matched_chunk is None:
                        matched_chunk = chunk
                else:
                    matched_chunk = chunk
                    break

        if source_path.startswith("db://"):
            chunk_id = "sql_execution"
            quote_snippet = (
                matched_chunk.content[:160].strip()
                if matched_chunk
                else default_sql_snippet
            )
        else:
            chunk_id = matched_chunk.chunk_id if matched_chunk else ""
            quote_snippet = (
                matched_chunk.content[:160].strip() if matched_chunk else ""
            )

        dedup_key = (source_path, heading_path, chunk_id)
        if dedup_key not in seen:
            seen.add(dedup_key)
            citations.append(
                Citation(
                    source_path=source_path,
                    heading_path=heading_path,
                    chunk_id=chunk_id,
                    quote_snippet=quote_snippet,
                )
            )

    return citations


# ============================================================================
# 3. Prompt Definitions & Versioned Registry
# ============================================================================


@dataclass
class PromptDefinition:
    """Prompt template definition with local fallback and metadata for Langfuse compatibility."""

    name: str
    version: str
    system_template: str
    user_template: str = "{query}"
    description: str = ""
    input_variables: list[str] = field(default_factory=list)


# Local fallback prompt templates compatible with Langfuse Prompt Management

PROMPT_INTENT_ROUTER = PromptDefinition(
    name="intent_router",
    version="1.1.0",
    description="Classifies user query intent into chitchat, knowledge_query, structured_data_query, or clarify.",
    system_template=(
        "You are an expert intent classifier for CortexGraph, an agentic personal knowledge and database assistant.\n"
        "Analyze the user's message and ongoing conversation context to classify their intent into exactly one of:\n\n"
        "1. 'knowledge_query':\n"
        "   - The user asks for conceptual, qualitative, or textual information from personal notes, obsidian vaults, "
        "architecture decisions (ADRs), meeting minutes, postmortems, design docs, or research summaries.\n"
        "   - Examples:\n"
        "     * 'What did we decide about using SQLite vs Postgres in ADR-003?'\n"
        "     * 'Summarize my meeting notes on the authentication refactor from last week.'\n"
        "     * 'How does CRAG handle retrieval failure according to our architecture notes?'\n"
        "     * 'What are the core design principles in my obsidian vault?'\n\n"
        "2. 'structured_data_query':\n"
        "   - The user asks numerical, analytical, aggregate, relational, or transactional questions about enterprise "
        "business data, database tables, sales figures, revenue, inventory levels, order histories, stock items, customer "
        "records, or operational metrics from the Wide World Importers (wideworldimporters) relational database.\n"
        "   - Examples:\n"
        "     * 'What were our total sales numbers for Q3 across all territories?'\n"
        "     * 'Show me the top 5 customers by order volume this year.'\n"
        "     * 'How many items of \"USB Powered Desk Fan\" do we currently have in stock in inventory?'\n"
        "     * 'List the recent orders placed by Tailspin Toys and their shipment status.'\n"
        "     * 'What is the average order value and total revenue per customer category in the database?'\n"
        "     * 'Which suppliers delivered the most purchase orders last month?'\n\n"
        "3. 'chitchat':\n"
        "   - Greetings, casual conversational turns, pleasantries, thanking, or inquiries about your identity.\n"
        "   - Examples:\n"
        "     * 'Hello! How are you doing today?'\n"
        "     * 'Who created you and what can you help me with?'\n"
        "     * 'Thanks for the help, that was awesome!'\n\n"
        "4. 'clarify':\n"
        "   - The user query is too ambiguous, severely underspecified, unintelligible, or missing critical context required "
        "to determine whether it refers to notes or database tables.\n"
        "   - Examples:\n"
        "     * 'Tell me about the thing.'\n"
        "     * 'Show me more.'\n"
        "     * 'What was that number again?' (without prior conversation context)\n\n"
        "Guidelines:\n"
        "- If a query asks for specific numbers, counts, aggregates, financial figures, customer orders, or inventory items "
        "from database tables, ALWAYS classify as 'structured_data_query'.\n"
        "- If a query asks about technical concepts, personal thoughts, design rationale, or markdown notes, classify as 'knowledge_query'.\n\n"
        "Provide your classification and concise reasoning."
    ),
    user_template="Conversation Summary: {summary}\nUser Message: {query}",
    input_variables=["summary", "query"],
)

# Export alias matching issue naming convention
INTENT_ROUTER_PROMPT = PROMPT_INTENT_ROUTER

PROMPT_QUERY_ANALYZER = PromptDefinition(
    name="query_analyzer",
    version="1.0.0",
    description="Decontextualizes queries and extracts metadata filters for vector retrieval.",
    system_template=(
        "You are an expert query analyzer for a personal knowledge assistant.\n"
        "Given the user's latest query, conversation history, and summary:\n"
        "1. Generate a self-contained, decontextualized 'standalone_query' that resolves pronouns (it, that, they), "
        "conversational anaphora, and context into explicit terminology suitable for semantic vector search.\n"
        "2. Extract any explicit metadata filters requested by the user:\n"
        "   - 'source_type_filter': e.g., 'markdown', 'pdf', or null if unspecified.\n"
        "   - 'created_after_ts': Unix timestamp if a timeframe is specified (e.g., 'notes from this week'), or null.\n"
        "   - 'required_tags': Specific tags mentioned (e.g., 'architecture', 'langgraph', 'adr'), or empty list."
    ),
    user_template=(
        "Conversation Summary: {summary}\n"
        "Active Filters: {metadata_filters}\n"
        "User Query: {query}"
    ),
    input_variables=["summary", "metadata_filters", "query"],
)

PROMPT_DOCUMENT_GRADER = PromptDefinition(
    name="document_grader",
    version="1.0.0",
    description="Evaluates the relevance of a retrieved knowledge chunk against the user query.",
    system_template=(
        "You are an expert relevance evaluator for a Corrective RAG (CRAG) system.\n"
        "Your task is to assess whether a retrieved document chunk from the user's personal knowledge vault "
        "contains information or context relevant to answering the user's query.\n\n"
        "Guidelines:\n"
        "- Grade 'is_relevant: true' if the chunk contains facts, explanations, code, or context useful for the query.\n"
        "- Grade 'is_relevant: false' if the chunk is entirely off-topic, tangential, or does not help answer the query.\n"
        "- Be objective and concise in your reasoning."
    ),
    user_template=(
        "User Query: {query}\n\n"
        "Retrieved Chunk [ID: {chunk_id}]\n"
        "Source: {source_path}\n"
        "Heading: {heading_path}\n"
        "Content:\n{content}\n\n"
        "Evaluate relevance for chunk {chunk_id}:"
    ),
    input_variables=["query", "chunk_id", "source_path", "heading_path", "content"],
)

PROMPT_QUERY_REWRITER = PromptDefinition(
    name="query_rewriter",
    version="1.0.0",
    description="Reformulates search query and relaxes restrictive metadata filters after retrieval failure.",
    system_template=(
        "You are an expert retrieval optimizer for a Corrective RAG system.\n"
        "The previous query failed to retrieve relevant documents from the user's personal knowledge vault.\n"
        "Analyze the original query, failed query, conversation context, and current metadata filters.\n\n"
        "Tasks:\n"
        "1. Reformulate the search query into 'rewritten_query' using alternative synonyms, broader technical concepts, "
        "related keywords, or simplified phrasing.\n"
        "2. Review active metadata filters. If filters (such as strict tags or source_type) might have caused 0 results, "
        "relax or remove them by setting 'relax_filters: true' and providing broadened or cleared filter values."
    ),
    user_template=(
        "Conversation Summary: {summary}\n"
        "Original Query: {original_query}\n"
        "Failed Query: {failed_query}\n"
        "Rewrite Attempt: {rewrite_count}\n"
        "Current Filters: {metadata_filters}\n\n"
        "Generate reformulated query and filter adjustments:"
    ),
    input_variables=["summary", "original_query", "failed_query", "rewrite_count", "metadata_filters"],
)

PROMPT_RAG_SYNTHESIZER = PromptDefinition(
    name="rag_synthesizer",
    version="1.0.0",
    description="Synthesizes grounded answers strictly from retrieved notes with mandatory inline citations.",
    system_template=(
        "You are CortexGraph, an AI second brain and personal knowledge assistant.\n"
        "Answer the user's query EXCLUSIVELY using the facts contained within the verified note chunks below.\n\n"
        "STRICT GROUNDING & CITATION RULES:\n"
        "1. Groundedness: Answer ONLY using facts explicitly present in the provided notes. Never speculate, guess, "
        "hallucinate, or introduce outside pre-training knowledge.\n"
        "2. Mandatory Inline Citations: Every factual claim, statement, or technical detail MUST be immediately followed "
        "by an inline citation in the exact format: `[Source: {source_path}#{heading_path}]`.\n"
        "   - If heading_path is empty, use `[Source: {source_path}]`.\n"
        "   - Do NOT alter or translate the source_path or heading_path.\n"
        "3. Abstention Guardrail: If the provided notes do not contain sufficient facts to answer the question, or if "
        "no relevant chunks exist, explicitly state what information is missing from the personal knowledge base "
        "rather than guessing from pre-training knowledge.\n"
        "4. Tone: Clear, concise, authoritative, structured markdown."
    ),
    user_template=(
        "Verified Knowledge Vault Chunks:\n"
        "{context}\n\n"
        "User Query: {query}\n\n"
        "Answer with strict inline citations:"
    ),
    input_variables=["context", "query"],
)

PROMPT_HALLUCINATION_GRADER = PromptDefinition(
    name="hallucination_grader",
    version="1.0.0",
    description="Grades factual groundedness and query relevance of synthesized answers.",
    system_template=(
        "You are an expert fact-checking and groundedness judge for a Corrective RAG system.\n"
        "Compare the assistant's generated response against the provided reference knowledge chunks.\n\n"
        "Evaluation Criteria:\n"
        "1. Groundedness ('is_grounded'): Are all factual claims, numbers, code examples, and assertions in the answer "
        "strictly supported by the provided reference chunks? If the response invents facts not in the source text, "
        "grade 'is_grounded: false'. If the answer explicitly states that information is missing (clean abstention), "
        "grade 'is_grounded: true'.\n"
        "2. Query Relevance ('addresses_query'): Does the response directly address what the user asked?\n"
        "3. Critique: If ungrounded or incomplete, explain what facts are unsupported."
    ),
    user_template=(
        "Reference Knowledge Chunks:\n"
        "{context}\n\n"
        "User Query: {query}\n\n"
        "Assistant Response:\n"
        "{response}\n\n"
        "Evaluate groundedness and query satisfaction:"
    ),
    input_variables=["context", "query", "response"],
)

PROMPT_DIRECT_RESPONDER = PromptDefinition(
    name="direct_responder",
    version="1.0.0",
    description="Generates direct conversational responses for chitchat or clarification requests.",
    system_template=(
        "You are CortexGraph, an AI second brain and personal knowledge assistant built on LangGraph, ChromaDB, and OpenAI.\n"
        "Mode: {mode}\n\n"
        "Guidelines:\n"
        "- If mode is 'chitchat': Provide a warm, concise, and helpful conversational reply. Introduce your capabilities "
        "as a personal knowledge assistant when appropriate.\n"
        "- If mode is 'clarify': Politely explain why the request is ambiguous or what specific information is needed "
        "so the user can clarify their query."
    ),
    user_template="Conversation Summary: {summary}\nUser Message: {query}",
    input_variables=["mode", "summary", "query"],
)

# Registry mapping template names to local fallback definitions
PROMPT_REGISTRY: dict[str, PromptDefinition] = {
    "intent_router": PROMPT_INTENT_ROUTER,
    "query_analyzer": PROMPT_QUERY_ANALYZER,
    "document_grader": PROMPT_DOCUMENT_GRADER,
    "query_rewriter": PROMPT_QUERY_REWRITER,
    "rag_synthesizer": PROMPT_RAG_SYNTHESIZER,
    "hallucination_grader": PROMPT_HALLUCINATION_GRADER,
    "direct_responder": PROMPT_DIRECT_RESPONDER,
}


def get_prompt_definition(name: str) -> PromptDefinition:
    """Retrieve the prompt definition for the given name from the local registry."""
    if name not in PROMPT_REGISTRY:
        raise KeyError(f"Prompt '{name}' not found in PROMPT_REGISTRY. Available: {list(PROMPT_REGISTRY.keys())}")
    return PROMPT_REGISTRY[name]


def get_prompt(
    name: str,
    version: str | None = None,
    fallback: str | None = None,
    langfuse_client: Any | None = None,
) -> str:
    """Retrieve system prompt template with Langfuse synchronization and local fallback.

    If Langfuse client is provided and configured, attempts remote prompt fetch.
    Gracefully falls back to local definitions on any error or missing client.

    Args:
        name: Name of the prompt in the registry.
        version: Optional prompt version identifier.
        fallback: Optional manual fallback string override.
        langfuse_client: Optional Langfuse client instance.

    Returns:
        System prompt template string.
    """
    if langfuse_client is not None:
        try:
            # Langfuse prompt management integration
            remote_prompt = langfuse_client.get_prompt(name=name, version=version)
            if remote_prompt and hasattr(remote_prompt, "prompt"):
                return str(remote_prompt.prompt)
        except Exception as exc:
            logger.debug("Langfuse prompt fetch for '%s' failed, using local fallback: %s", name, exc)

    if fallback is not None:
        return fallback

    definition = get_prompt_definition(name)
    return definition.system_template


def format_prompt(name: str, **kwargs: Any) -> tuple[str, str]:
    """Format both system and user templates for a registered prompt.

    Args:
        name: Registered prompt name.
        **kwargs: Substitution variables for system and user templates.

    Returns:
        Tuple of (formatted_system_prompt, formatted_user_prompt).
    """
    definition = get_prompt_definition(name)
    sys_tmpl = definition.system_template
    usr_tmpl = definition.user_template

    # Safe formatting allowing missing optional keys
    for k, v in kwargs.items():
        placeholder = f"{{{k}}}"
        sys_tmpl = sys_tmpl.replace(placeholder, str(v))
        usr_tmpl = usr_tmpl.replace(placeholder, str(v))

    return sys_tmpl, usr_tmpl
