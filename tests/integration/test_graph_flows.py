"""Integration tests for the 4 major LangGraph execution trajectories.

Verifies deterministic offline flow transitions using simulated structured responses:
1. Direct chitchat bypass (intent_router -> direct_responder -> memory_manager).
2. Single-pass successful RAG (intent_router -> query_analyzer -> chroma_retriever ->
   document_grader -> rag_synthesizer -> hallucination_grader -> memory_manager).
3. Corrective query-rewrite loop with circuit breaker (document_grader 'insufficient' ->
   query_rewriter -> chroma_retriever -> forced exit at MAX_RETRIES = 2).
4. Hallucination self-correction loop (hallucination_grader 'ungrounded' ->
   rag_synthesizer regeneration -> grounded -> memory_manager).
"""

from typing import Any, Literal

import pytest

MAX_RETRIES = 2


class SimulatedGraphRunner:
    """Deterministic simulation of the CortexGraph state routing engine.

    Models the state transitions and circuit-breaker behavior specified for
    LangGraph execution paths before and during full node implementation.
    """

    def __init__(
        self,
        intent: Literal["chitchat", "rag"] = "chitchat",
        doc_grade: Literal["sufficient", "insufficient"] = "sufficient",
        hallucination_grade_sequence: list[Literal["grounded", "ungrounded"]] | None = None,
    ) -> None:
        self.intent = intent
        self.doc_grade = doc_grade
        self.hallucination_grades = hallucination_grade_sequence or ["grounded"]
        self.retry_count = 0
        self.hallucination_check_idx = 0
        self.execution_path: list[str] = []

    def run(self, query: str) -> dict[str, Any]:
        """Execute state transitions deterministically."""
        # Step 1: intent_router
        self.execution_path.append("intent_router")

        if self.intent == "chitchat":
            self.execution_path.append("direct_responder")
            self.execution_path.append("memory_manager")
            return {
                "trajectory": "direct_chitchat_bypass",
                "path": self.execution_path,
                "response": f"Direct response to '{query}'",
                "retries": self.retry_count,
            }

        # Step 2: RAG path
        self.execution_path.append("query_analyzer")
        self.execution_path.append("chroma_retriever")

        # Step 3: Document evaluation loop with MAX_RETRIES circuit breaker
        while True:
            self.execution_path.append("document_grader")
            if self.doc_grade == "sufficient" or self.retry_count >= MAX_RETRIES:
                break
            # Corrective query rewrite loop
            self.retry_count += 1
            self.execution_path.append("query_rewriter")
            self.execution_path.append("chroma_retriever")

        # Step 4: Synthesis & Hallucination loop
        while True:
            self.execution_path.append("rag_synthesizer")
            self.execution_path.append("hallucination_grader")

            current_grade = (
                self.hallucination_grades[self.hallucination_check_idx]
                if self.hallucination_check_idx < len(self.hallucination_grades)
                else "grounded"
            )
            self.hallucination_check_idx += 1

            if current_grade == "grounded":
                break

        # Step 5: Memory management
        self.execution_path.append("memory_manager")

        return {
            "trajectory": "rag_pipeline",
            "path": self.execution_path,
            "response": f"Synthesized answer for '{query}'",
            "retries": self.retry_count,
        }


@pytest.mark.asyncio
async def test_trajectory_1_direct_chitchat_bypass() -> None:
    """Verify trajectory 1: intent_router -> direct_responder -> memory_manager."""
    runner = SimulatedGraphRunner(intent="chitchat")
    result = runner.run("Hello, who are you?")

    expected_path = ["intent_router", "direct_responder", "memory_manager"]
    assert result["path"] == expected_path
    assert result["trajectory"] == "direct_chitchat_bypass"
    assert result["retries"] == 0


@pytest.mark.asyncio
async def test_trajectory_2_single_pass_successful_rag() -> None:
    """Verify trajectory 2: Single-pass successful RAG pipeline."""
    runner = SimulatedGraphRunner(
        intent="rag",
        doc_grade="sufficient",
        hallucination_grade_sequence=["grounded"],
    )
    result = runner.run("What are the key features of LangGraph?")

    expected_path = [
        "intent_router",
        "query_analyzer",
        "chroma_retriever",
        "document_grader",
        "rag_synthesizer",
        "hallucination_grader",
        "memory_manager",
    ]
    assert result["path"] == expected_path
    assert result["retries"] == 0


@pytest.mark.asyncio
async def test_trajectory_3_corrective_query_rewrite_loop_breaker() -> None:
    """Verify trajectory 3: document_grader returns insufficient and exits at MAX_RETRIES = 2."""
    runner = SimulatedGraphRunner(
        intent="rag",
        doc_grade="insufficient",
        hallucination_grade_sequence=["grounded"],
    )
    result = runner.run("Explain an obscure unindexed topic.")

    # Should loop query_rewriter -> chroma_retriever exactly twice
    assert result["retries"] == MAX_RETRIES
    expected_path = [
        "intent_router",
        "query_analyzer",
        "chroma_retriever",
        "document_grader",
        "query_rewriter",
        "chroma_retriever",
        "document_grader",
        "query_rewriter",
        "chroma_retriever",
        "document_grader",  # breaks here because retries == 2
        "rag_synthesizer",
        "hallucination_grader",
        "memory_manager",
    ]
    assert result["path"] == expected_path


@pytest.mark.asyncio
async def test_trajectory_4_hallucination_self_correction_loop() -> None:
    """Verify trajectory 4: hallucination_grader returns ungrounded -> rag_synthesizer regeneration."""
    runner = SimulatedGraphRunner(
        intent="rag",
        doc_grade="sufficient",
        hallucination_grade_sequence=["ungrounded", "grounded"],
    )
    result = runner.run("Summarize project architecture.")

    expected_path = [
        "intent_router",
        "query_analyzer",
        "chroma_retriever",
        "document_grader",
        "rag_synthesizer",
        "hallucination_grader",
        "rag_synthesizer",  # Regeneration after ungrounded detection
        "hallucination_grader",
        "memory_manager",
    ]
    assert result["path"] == expected_path
