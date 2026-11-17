"""Unit tests for structure-aware document parser and two-stage chunking pipeline."""

from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.rag.parser import (
    MarkdownASTParser,
    ParsedDocument,
    calculate_doc_hash,
    calculate_file_hash,
    count_tokens,
    extract_frontmatter,
    extract_wikilinks,
    normalize_tags,
    normalize_wikilinks,
    parse_created_timestamp,
    parse_file,
    parse_file_to_document,
    parse_markdown,
    parse_pdf,
    parse_text,
)


class TestFrontmatterExtractor:
    """Tests for YAML frontmatter extraction and parsing."""

    def test_extract_standard_frontmatter(self) -> None:
        content = """---
title: "Autonomous Agent Architecture"
tags: [ai, rag, langgraph]
created: 2026-10-14 09:30:00
aliases: ["AutoAgent", "AgentArch"]
---
# Introduction
This is the note body.
"""
        frontmatter, body = extract_frontmatter(content)
        assert frontmatter["title"] == "Autonomous Agent Architecture"
        assert frontmatter["tags"] == ["ai", "rag", "langgraph"]
        assert frontmatter["aliases"] == ["AutoAgent", "AgentArch"]
        assert body.strip() == "# Introduction\nThis is the note body."

    def test_extract_missing_frontmatter(self) -> None:
        content = "# Just a Note\nNo frontmatter here."
        frontmatter, body = extract_frontmatter(content)
        assert frontmatter == {}
        assert body == content

    def test_extract_malformed_frontmatter(self) -> None:
        content = """---
title: [unclosed list
---
# Broken YAML
"""
        frontmatter, body = extract_frontmatter(content)
        assert frontmatter == {}
        assert body == content


class TestTagsNormalization:
    """Tests for normalizing tags to ',tag1,tag2,' format for ChromaDB queries."""

    def test_normalize_list_tags(self) -> None:
        tags = ["#ai", "RAG", "langgraph"]
        assert normalize_tags(tags) == ",ai,rag,langgraph,"

    def test_normalize_comma_string_tags(self) -> None:
        tags = "ai, #rag, langgraph"
        assert normalize_tags(tags) == ",ai,rag,langgraph,"

    def test_normalize_space_string_tags(self) -> None:
        tags = "#ai #rag #agent"
        assert normalize_tags(tags) == ",ai,rag,agent,"

    def test_normalize_single_tag(self) -> None:
        assert normalize_tags("personal-knowledge") == ",personal-knowledge,"

    def test_normalize_duplicates_and_whitespace(self) -> None:
        tags = ["  AI  ", "ai", "#AI", "vector-db"]
        assert normalize_tags(tags) == ",ai,vector-db,"

    def test_normalize_empty_tags(self) -> None:
        assert normalize_tags([]) == ""
        assert normalize_tags("") == ""
        assert normalize_tags(None) == ""


class TestTimestampParsing:
    """Tests for parsing various timestamp representations into Unix timestamps."""

    def test_parse_iso8601_string(self) -> None:
        ts = parse_created_timestamp("2026-10-14T09:00:00Z")
        assert isinstance(ts, int)
        assert ts > 0

    def test_parse_date_string(self) -> None:
        ts = parse_created_timestamp("2026-10-14")
        assert isinstance(ts, int)
        assert ts > 0

    def test_parse_datetime_object(self) -> None:
        dt = datetime.datetime(2026, 10, 14, 9, 0, 0, tzinfo=datetime.UTC)
        assert parse_created_timestamp(dt) == int(dt.timestamp())

    def test_parse_date_object(self) -> None:
        d = datetime.date(2026, 10, 14)
        ts = parse_created_timestamp(d)
        assert isinstance(ts, int)
        assert ts > 0

    def test_parse_numeric_timestamp(self) -> None:
        assert parse_created_timestamp(1728896400) == 1728896400
        assert parse_created_timestamp("1728896400") == 1728896400

    def test_parse_fallback_on_invalid(self) -> None:
        assert parse_created_timestamp("invalid-date", fallback_ts=42) == 42
        assert parse_created_timestamp(None, fallback_ts=100) == 100


class TestWikiLinkNormalizer:
    """Tests for Obsidian/Logseq [[WikiLink]] normalization and extraction."""

    def test_normalize_plain_wikilink(self) -> None:
        text = "See [[Machine Learning]] for background."
        assert normalize_wikilinks(text) == "See Machine Learning for background."

    def test_normalize_piped_wikilink(self) -> None:
        text = "Refer to [[Machine Learning|ML]] overview."
        assert normalize_wikilinks(text) == "Refer to ML overview."

    def test_normalize_heading_wikilink(self) -> None:
        text = "Detailed in [[Architecture#Database]]."
        assert normalize_wikilinks(text) == "Detailed in Architecture (Database)."

    def test_normalize_heading_with_display_wikilink(self) -> None:
        text = "Check [[Architecture#Database|Postgres Configuration]]."
        assert normalize_wikilinks(text) == "Check Postgres Configuration."

    def test_extract_wikilinks(self) -> None:
        text = "Links to [[Note A]], [[Note B|Alias B]], and [[Note A#Heading]]."
        extracted = extract_wikilinks(text)
        assert extracted == ["Note A", "Note B"]


class TestDocHashCalculations:
    """Tests for SHA-256 hash calculation, determinism, and whitespace idempotency."""

    def test_whitespace_normalization_idempotency(self) -> None:
        doc_crlf = "---  \r\ntitle: Architecture  \r\n---\r\n\r\n# Overview  \r\nSome text.   \r\n\r\n"
        doc_lf = "---\ntitle: Architecture\n---\n\n# Overview\nSome text.\n"
        doc_spaces = "--- \ntitle: Architecture \n---\n\n# Overview \nSome text. \n"

        hash_crlf = calculate_doc_hash(doc_crlf)
        hash_lf = calculate_doc_hash(doc_lf)
        hash_spaces = calculate_doc_hash(doc_spaces)

        assert hash_crlf == hash_lf, "CRLF and LF differences must yield identical hashes"
        assert hash_lf == hash_spaces, "Trailing line whitespace must yield identical hashes"

    def test_content_change_modifies_hash(self) -> None:
        doc1 = "# Overview\nSystem uses Postgres for persistence."
        doc2 = "# Overview\nSystem uses SQLite for persistence."

        hash1 = calculate_doc_hash(doc1)
        hash2 = calculate_doc_hash(doc2)

        assert hash1 != hash2, "Content modifications must deterministically alter doc_hash"
        assert len(hash1) == 64
        assert len(hash2) == 64

    def test_calculate_file_hash_on_disk(self, tmp_path: Path) -> None:
        file1 = tmp_path / "note1.md"
        file2 = tmp_path / "note2.md"

        file1.write_bytes(b"# Test Note\nLine 1   \n")
        file2.write_bytes(b"# Test Note\r\nLine 1\r\n\r\n")

        assert calculate_file_hash(file1) == calculate_file_hash(file2)


class TestTwoStageStructureAwareChunking:
    """Tests for Stage 1 (AST Header Split) and Stage 2 (Recursive Token Splitter)."""

    def test_heading_breadcrumb_formation(self) -> None:
        markdown = """# CortexGraph
Top level project overview.

## Architecture
High-level system components.

### Vector Store
ChromaDB integration details.

### Graph Engine
LangGraph state transitions.

# Deployment
Docker compose configuration.
"""
        chunks = parse_markdown(markdown, source_path="docs/architecture.md")
        assert len(chunks) >= 5

        heading_paths = [c.heading_path for c in chunks]
        assert "CortexGraph" in heading_paths
        assert "CortexGraph > Architecture" in heading_paths
        assert "CortexGraph > Architecture > Vector Store" in heading_paths
        assert "CortexGraph > Architecture > Graph Engine" in heading_paths
        assert "Deployment" in heading_paths

    def test_chunk_header_prepended(self) -> None:
        markdown = """# Memory Manager
Implements rolling summarization of conversational context.
"""
        chunks = parse_markdown(markdown, source_path="notes/memory.md")
        assert len(chunks) == 1
        expected_header = "[File: notes/memory.md | Section: Memory Manager]\n"
        assert chunks[0].content.startswith(expected_header)
        assert chunks[0].source_path == "notes/memory.md"
        assert chunks[0].heading_path == "Memory Manager"

    def test_max_token_boundaries(self) -> None:
        # Create a large section exceeding 512 tokens (~1500 tokens)
        long_paragraph = " ".join(["knowledge"] * 1200)
        markdown = f"""# Massive Document
## Detailed Specifications
{long_paragraph}
"""
        chunks = parse_markdown(
            markdown,
            source_path="vault/spec.md",
            chunk_size=512,
            chunk_overlap=64,
        )

        assert len(chunks) > 1, "Large section should be split into multiple sub-chunks"

        for idx, chunk in enumerate(chunks):
            tokens = count_tokens(chunk.content)
            assert (
                tokens <= 512
            ), f"Chunk {idx} token count ({tokens}) exceeds maximum boundary of 512 tokens"
            assert chunk.content.startswith(
                "[File: vault/spec.md | Section: Massive Document > Detailed Specifications]\n"
            )
            assert chunk.metadata["chunk_index"] == idx
            assert chunk.metadata["total_chunks"] == len(chunks)
            assert chunk.metadata["source_type"] == "markdown"

    def test_markdown_without_headings(self) -> None:
        markdown = "Simple thought captured without any markdown headings."
        chunks = parse_markdown(markdown, source_path="inbox/thought.md")
        assert len(chunks) == 1
        assert chunks[0].heading_path == ""
        assert chunks[0].content.startswith("[File: inbox/thought.md | Section: ]\n")

    def test_frontmatter_metadata_in_chunks(self) -> None:
        markdown = """---
title: "Cyclic Graph ADR"
tags: [architecture, langgraph, adr]
created: "2026-10-14T10:00:00Z"
aliases: [ADR-004]
---
# Decision
We adopt LangGraph with MAX_RETRIES=2 circuit breakers.
"""
        chunks = parse_markdown(markdown, source_path="adrs/004.md")
        assert len(chunks) == 1
        meta = chunks[0].metadata

        assert meta["title"] == "Cyclic Graph ADR"
        assert meta["tags"] == ",architecture,langgraph,adr,"
        assert isinstance(meta["created_at_ts"], int)
        assert meta["created_at_ts"] > 0
        assert meta["aliases"] == "ADR-004"
        assert meta["source_type"] == "markdown"


class TestMultiFormatParsers:
    """Tests for parsing .txt, .py, .json, and .pdf files."""

    def test_parse_text_file(self) -> None:
        content = "Plain text notes line 1.\nLine 2.\nLine 3."
        chunks = parse_text(content, source_path="notes.txt", source_type="text")
        assert len(chunks) == 1
        assert chunks[0].metadata["source_type"] == "text"
        assert chunks[0].content.startswith("[File: notes.txt | Section: ]\n")

    def test_parse_python_file(self, tmp_path: Path) -> None:
        py_file = tmp_path / "sample.py"
        py_file.write_text("def hello() -> str:\n    return 'world'\n", encoding="utf-8")

        chunks = parse_file(py_file)
        assert len(chunks) == 1
        assert chunks[0].metadata["source_type"] == "python"
        assert chunks[0].metadata["title"] == "sample"

    def test_parse_json_file(self, tmp_path: Path) -> None:
        json_file = tmp_path / "config.json"
        json_file.write_text('{"name": "cortex", "retries": 2}', encoding="utf-8")

        chunks = parse_file(json_file)
        assert len(chunks) == 1
        assert chunks[0].metadata["source_type"] == "json"
        assert '"retries": 2' in chunks[0].content

    def test_parse_pdf_mocked(self, tmp_path: Path) -> None:
        pdf_file = tmp_path / "whitepaper.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy content")

        mock_page_1 = MagicMock()
        mock_page_1.extract_text.return_value = "Page 1 Content: Executive Summary."
        mock_page_2 = MagicMock()
        mock_page_2.extract_text.return_value = "Page 2 Content: Technical Evaluation."

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page_1, mock_page_2]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            chunks = parse_pdf(pdf_file)
            assert len(chunks) == 2
            assert chunks[0].heading_path == "Page 1"
            assert chunks[0].content.startswith(
                f"[File: {pdf_file.as_posix()} | Section: Page 1]\n"
            )
            assert chunks[1].heading_path == "Page 2"
            assert chunks[1].content.startswith(
                f"[File: {pdf_file.as_posix()} | Section: Page 2]\n"
            )
            assert chunks[0].metadata["source_type"] == "pdf"


class TestParsedDocumentAndParserClass:
    """Tests for ParsedDocument model and MarkdownASTParser class."""

    def test_parse_file_to_document(self, tmp_path: Path) -> None:
        note_file = tmp_path / "architecture.md"
        note_file.write_text(
            """---
title: System Overview
tags: [core]
---
# Core Design
System components.
""",
            encoding="utf-8",
        )

        doc = parse_file_to_document(note_file)
        assert isinstance(doc, ParsedDocument)
        assert doc.title == "System Overview"
        assert doc.tags == ",core,"
        assert doc.source_type == "markdown"
        assert len(doc.chunks) == 1
        assert doc.chunks[0].heading_path == "Core Design"

    def test_markdown_ast_parser_class_interface(self) -> None:
        parser = MarkdownASTParser(chunk_size=256, chunk_overlap=32)
        markdown = """# Section 1
Content here.
## Section 2
More content.
"""
        chunks = parser.parse_markdown(markdown, source_path="test.md")
        assert len(chunks) == 2
        assert chunks[0].heading_path == "Section 1"
        assert chunks[1].heading_path == "Section 1 > Section 2"

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        bin_file = tmp_path / "binary.bin"
        bin_file.write_bytes(b"\x00\x01\x02")

        with pytest.raises(ValueError, match="Unsupported file format"):
            parse_file(bin_file)

    def test_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            parse_file("nonexistent_file.md")
