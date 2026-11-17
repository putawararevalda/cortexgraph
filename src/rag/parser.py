"""Structure-aware document parsing and two-stage chunking pipeline for CortexGraph.

Converts raw personal notes (Obsidian/Logseq Markdown vaults, PDFs, code, and text notes)
into context-rich chunks ready for ChromaDB indexing.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import dateutil.parser
import tiktoken
import yaml
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pydantic import BaseModel, ConfigDict, Field

from src.graph.state import RetrievedChunk

# Regular expressions
FRONTMATTER_REGEX = re.compile(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)
WIKILINK_REGEX = re.compile(r"\[\[([^\]]+)\]\]")

# Default headers to split on for Markdown Stage 1 AST splitting
DEFAULT_HEADERS_TO_SPLIT_ON: list[tuple[str, str]] = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
]

# Supported file extensions mapped to source types
EXTENSION_SOURCE_TYPE_MAP: dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".pdf": "pdf",
    ".txt": "text",
    ".py": "python",
    ".json": "json",
}


class ParsedDocument(BaseModel):
    """Structured representation of a parsed document and its generated chunks."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    source_path: str
    doc_hash: str
    source_type: str
    raw_content: str
    title: str = ""
    tags: str = ""
    created_at_ts: int = 0
    frontmatter: dict[str, Any] = Field(default_factory=dict)
    chunks: list[RetrievedChunk] = Field(default_factory=list)


def get_token_encoder(model_or_encoding: str = "cl100k_base") -> tiktoken.Encoding:
    """Retrieve or cache the tiktoken encoding instance."""
    try:
        return tiktoken.get_encoding(model_or_encoding)
    except ValueError:
        return tiktoken.encoding_for_model(model_or_encoding)


def count_tokens(text: str, model_or_encoding: str = "cl100k_base") -> int:
    """Count tokens in a string using tiktoken."""
    encoder = get_token_encoder(model_or_encoding)
    return len(encoder.encode(text))


def normalize_content_for_hashing(content: str) -> str:
    """Normalize text content so whitespace variations remain idempotent for SHA-256 hashing.

    Normalizes Windows/Mac line endings to LF, strips trailing line whitespace,
    and removes leading/trailing blank lines.
    """
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def calculate_doc_hash(content: str) -> str:
    """Compute deterministic SHA-256 hash of whitespace-normalized document content."""
    normalized = normalize_content_for_hashing(content)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def calculate_file_hash(file_path: str | Path) -> str:
    """Calculate deterministic SHA-256 hash for a file on disk.

    Text-based files (.md, .txt, .py, .json) are whitespace-normalized before hashing.
    PDFs and binary files are hashed via normalized text if extractable, or raw bytes.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in {".md", ".markdown", ".txt", ".py", ".json"}:
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="utf-8", errors="replace")
        return calculate_doc_hash(content)

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            pages_text = [page.extract_text() or "" for page in reader.pages]
            combined = "\n".join(pages_text).strip()
            if combined:
                return calculate_doc_hash(combined)
        except Exception:
            pass

    # Fallback to byte hashing
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Extract YAML frontmatter and remaining body from Markdown content.

    Returns:
        tuple[dict[str, Any], str]: (parsed_frontmatter_dict, remaining_markdown_body)
    """
    match = FRONTMATTER_REGEX.match(content)
    if not match:
        return {}, content

    raw_yaml = match.group(1)
    body = content[match.end() :]

    try:
        parsed = yaml.safe_load(raw_yaml)
        if isinstance(parsed, dict):
            return parsed, body
        return {}, body
    except yaml.YAMLError:
        return {}, content


def normalize_tags(tags: Any) -> str:
    """Normalize tags into a comma-delimited string format (`,tag1,tag2,`).

    Compatible with ChromaDB substring/metadata queries (e.g., `,tag,`).
    Returns an empty string if no valid tags exist.
    """
    if not tags:
        return ""

    raw_items: list[str] = []
    if isinstance(tags, (list, tuple, set)):
        raw_items = [str(t) for t in tags]
    elif isinstance(tags, str):
        raw_items = tags.split(",") if "," in tags else tags.split()
    else:
        raw_items = [str(tags)]

    cleaned: list[str] = []
    for item in raw_items:
        tag = item.strip()
        if tag.startswith("#"):
            tag = tag.lstrip("#")
        tag = tag.strip().lower()
        if tag and tag not in cleaned:
            cleaned.append(tag)

    if not cleaned:
        return ""

    return f",{','.join(cleaned)},"


def parse_created_timestamp(value: Any, fallback_ts: int = 0) -> int:
    """Parse various timestamp formats into an integer Unix timestamp.

    Handles ISO 8601 strings, datetime/date objects, floats, and numeric strings.
    Falls back to fallback_ts if parsing fails.
    """
    if value is None or value == "":
        return fallback_ts

    if isinstance(value, (int, float)):
        return int(value)

    if isinstance(value, datetime.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=datetime.UTC)
        return int(value.timestamp())

    if isinstance(value, datetime.date):
        dt = datetime.datetime(value.year, value.month, value.day, tzinfo=datetime.UTC)
        return int(dt.timestamp())

    if isinstance(value, str):
        val_str = value.strip()
        try:
            return int(float(val_str))
        except ValueError:
            pass

        try:
            dt = dateutil.parser.parse(val_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.UTC)
            return int(dt.timestamp())
        except (ValueError, TypeError, OverflowError):
            pass

    return fallback_ts


def normalize_wikilinks(text: str) -> str:
    """Normalize Obsidian/Logseq [[WikiLinks]] into clean, natural text.

    Transformations:
      - [[Note Name]] -> Note Name
      - [[Note Name|Display Text]] -> Display Text
      - [[Note Name#Heading|Display Text]] -> Display Text
      - [[Note Name#Heading]] -> Note Name (Heading)
    """

    def _replace(match: re.Match[str]) -> str:
        content = match.group(1).strip()
        if "|" in content:
            _, display = content.split("|", 1)
            return display.strip()
        if "#" in content:
            target, section = content.split("#", 1)
            target = target.strip()
            section = section.strip()
            if target and section:
                return f"{target} ({section})"
            return target or section
        return content

    return WIKILINK_REGEX.sub(_replace, text)


def extract_wikilinks(text: str) -> list[str]:
    """Extract unique target note titles from [[WikiLinks]]."""
    links: list[str] = []
    for match in WIKILINK_REGEX.finditer(text):
        content = match.group(1).strip()
        target = content.split("|")[0].split("#")[0].strip()
        if target and target not in links:
            links.append(target)
    return links


def format_chunk_header(source_path: str, heading_path: str = "") -> str:
    """Generate the standardized context header prepended to each chunk."""
    return f"[File: {source_path} | Section: {heading_path}]\n"


def two_stage_chunk_markdown(
    content: str,
    source_path: str = "note.md",
    metadata: dict[str, Any] | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    doc_hash: str = "",
) -> list[RetrievedChunk]:
    """Execute two-stage structure-aware chunking on Markdown content.

    Stage 1: MarkdownHeaderTextSplitter splits on #, ##, ###, generating breadcrumb heading_path.
    Stage 2: RecursiveCharacterTextSplitter.from_tiktoken_encoder bounds sub-chunks to chunk_size tokens.

    Prepend `[File: {source_path} | Section: {heading_path}]\\n` to each chunk.
    Guarantees final chunk token boundary <= chunk_size (default 512 tokens).
    """
    base_meta = dict(metadata or {})
    if not doc_hash:
        doc_hash = base_meta.get("doc_hash") or calculate_doc_hash(content)

    # Stage 1: Structural AST Split
    headers_to_split = DEFAULT_HEADERS_TO_SPLIT_ON
    header_keys = [h[1] for h in headers_to_split]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split,
        strip_headers=True,
    )

    stage1_docs = markdown_splitter.split_text(content)
    if not stage1_docs:
        return []

    encoder = get_token_encoder()
    raw_chunks: list[dict[str, Any]] = []

    for doc in stage1_docs:
        section_text = doc.page_content.strip()
        if not section_text:
            continue

        # Form breadcrumb heading_path: e.g. "System Architecture > Database Layer"
        heading_parts = [doc.metadata[k] for k in header_keys if k in doc.metadata]
        heading_path = " > ".join(heading_parts)

        # Standard context header
        header = format_chunk_header(source_path=source_path, heading_path=heading_path)
        header_tokens = len(encoder.encode(header))

        # Stage 2: Recursive Token Splitter with effective chunk size accounting for header
        effective_chunk_size = max(64, chunk_size - header_tokens)
        effective_overlap = min(chunk_overlap, effective_chunk_size // 2)

        token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=effective_chunk_size,
            chunk_overlap=effective_overlap,
        )

        sub_texts = token_splitter.split_text(section_text)
        for sub_text in sub_texts:
            clean_sub = sub_text.strip()
            if not clean_sub:
                continue

            full_chunk_content = f"{header}{clean_sub}"
            total_tokens = len(encoder.encode(full_chunk_content))

            # Defensive check: if total_tokens exceeds chunk_size, further truncate if needed
            if total_tokens > chunk_size:
                # Truncate tokens to strictly enforce boundary
                tokens = encoder.encode(full_chunk_content)[:chunk_size]
                full_chunk_content = encoder.decode(tokens)
                total_tokens = len(tokens)

            raw_chunks.append(
                {
                    "content": full_chunk_content,
                    "heading_path": heading_path,
                    "token_count": total_tokens,
                    "char_count": len(full_chunk_content),
                }
            )

    total_chunks = len(raw_chunks)
    chunks: list[RetrievedChunk] = []

    for idx, item in enumerate(raw_chunks):
        chunk_id = f"{doc_hash[:16]}_{idx:04d}"
        chunk_metadata: dict[str, Any] = {
            **base_meta,
            "source_path": source_path,
            "heading_path": item["heading_path"],
            "doc_hash": doc_hash,
            "source_type": base_meta.get("source_type", "markdown"),
            "chunk_index": idx,
            "total_chunks": total_chunks,
            "token_count": item["token_count"],
            "char_count": item["char_count"],
        }

        chunk = RetrievedChunk(
            chunk_id=chunk_id,
            content=item["content"],
            source_path=source_path,
            heading_path=item["heading_path"],
            score=0.0,
            metadata=chunk_metadata,
        )
        chunks.append(chunk)

    return chunks


def parse_markdown(
    content: str,
    source_path: str = "note.md",
    file_created_at_ts: int | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[RetrievedChunk]:
    """Parse a Markdown string, extract frontmatter, normalize WikiLinks, and generate chunks."""
    doc_hash = calculate_doc_hash(content)
    frontmatter, body = extract_frontmatter(content)

    # Title extraction
    title = frontmatter.get("title")
    if not title:
        first_h1_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
        if first_h1_match:
            title = first_h1_match.group(1).strip()
        else:
            title = Path(source_path).stem or "Untitled"

    # Tags normalization
    tags_str = normalize_tags(frontmatter.get("tags"))

    # Timestamp extraction
    raw_created = frontmatter.get("created")
    fallback_ts = file_created_at_ts if file_created_at_ts is not None else 0
    created_at_ts = parse_created_timestamp(raw_created, fallback_ts=fallback_ts)

    # WikiLink normalization and extraction
    wikilinks = extract_wikilinks(body)
    normalized_body = normalize_wikilinks(body)

    metadata: dict[str, Any] = {
        "title": str(title),
        "tags": tags_str,
        "created_at_ts": created_at_ts,
        "source_type": "markdown",
        "doc_hash": doc_hash,
    }
    if wikilinks:
        metadata["wikilinks"] = ",".join(wikilinks)
    if "aliases" in frontmatter:
        raw_aliases = frontmatter["aliases"]
        if isinstance(raw_aliases, list):
            metadata["aliases"] = ",".join(str(a) for a in raw_aliases)
        else:
            metadata["aliases"] = str(raw_aliases)

    return two_stage_chunk_markdown(
        content=normalized_body,
        source_path=source_path,
        metadata=metadata,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        doc_hash=doc_hash,
    )


def parse_pdf(
    file_path: str | Path,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[RetrievedChunk]:
    """Parse a PDF file into chunks using pypdf."""
    from pypdf import PdfReader

    path = Path(file_path)
    source_path = str(path.as_posix())
    doc_hash = calculate_file_hash(path)

    fallback_ts = int(path.stat().st_ctime) if path.exists() else 0
    title = path.stem

    reader = PdfReader(str(path))
    encoder = get_token_encoder()

    raw_chunks: list[dict[str, Any]] = []

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        page_text = (page.extract_text() or "").strip()
        if not page_text:
            continue

        heading_path = f"Page {page_num}"
        header = format_chunk_header(source_path=source_path, heading_path=heading_path)
        header_tokens = len(encoder.encode(header))

        effective_chunk_size = max(64, chunk_size - header_tokens)
        effective_overlap = min(chunk_overlap, effective_chunk_size // 2)

        token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=effective_chunk_size,
            chunk_overlap=effective_overlap,
        )

        sub_texts = token_splitter.split_text(page_text)
        for sub_text in sub_texts:
            clean_sub = sub_text.strip()
            if not clean_sub:
                continue

            full_chunk_content = f"{header}{clean_sub}"
            total_tokens = len(encoder.encode(full_chunk_content))
            if total_tokens > chunk_size:
                tokens = encoder.encode(full_chunk_content)[:chunk_size]
                full_chunk_content = encoder.decode(tokens)
                total_tokens = len(tokens)

            raw_chunks.append(
                {
                    "content": full_chunk_content,
                    "heading_path": heading_path,
                    "token_count": total_tokens,
                    "char_count": len(full_chunk_content),
                }
            )

    total_chunks = len(raw_chunks)
    chunks: list[RetrievedChunk] = []

    for idx, item in enumerate(raw_chunks):
        chunk_id = f"{doc_hash[:16]}_{idx:04d}"
        chunk_metadata: dict[str, Any] = {
            "source_path": source_path,
            "heading_path": item["heading_path"],
            "doc_hash": doc_hash,
            "source_type": "pdf",
            "title": title,
            "tags": "",
            "created_at_ts": fallback_ts,
            "chunk_index": idx,
            "total_chunks": total_chunks,
            "token_count": item["token_count"],
            "char_count": item["char_count"],
        }

        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                content=item["content"],
                source_path=source_path,
                heading_path=item["heading_path"],
                score=0.0,
                metadata=chunk_metadata,
            )
        )

    return chunks


def parse_text(
    content: str,
    source_path: str = "note.txt",
    source_type: str = "text",
    file_created_at_ts: int | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[RetrievedChunk]:
    """Parse plain text, code, or JSON content into chunks."""
    doc_hash = calculate_doc_hash(content)
    encoder = get_token_encoder()
    created_at_ts = file_created_at_ts if file_created_at_ts is not None else 0
    title = Path(source_path).stem or "Untitled"

    header = format_chunk_header(source_path=source_path, heading_path="")
    header_tokens = len(encoder.encode(header))

    effective_chunk_size = max(64, chunk_size - header_tokens)
    effective_overlap = min(chunk_overlap, effective_chunk_size // 2)

    token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=effective_chunk_size,
        chunk_overlap=effective_overlap,
    )

    clean_content = content.strip()
    if not clean_content:
        return []

    sub_texts = token_splitter.split_text(clean_content)
    raw_chunks: list[dict[str, Any]] = []

    for sub_text in sub_texts:
        clean_sub = sub_text.strip()
        if not clean_sub:
            continue

        full_chunk_content = f"{header}{clean_sub}"
        total_tokens = len(encoder.encode(full_chunk_content))
        if total_tokens > chunk_size:
            tokens = encoder.encode(full_chunk_content)[:chunk_size]
            full_chunk_content = encoder.decode(tokens)
            total_tokens = len(tokens)

        raw_chunks.append(
            {
                "content": full_chunk_content,
                "token_count": total_tokens,
                "char_count": len(full_chunk_content),
            }
        )

    total_chunks = len(raw_chunks)
    chunks: list[RetrievedChunk] = []

    for idx, item in enumerate(raw_chunks):
        chunk_id = f"{doc_hash[:16]}_{idx:04d}"
        chunk_metadata: dict[str, Any] = {
            "source_path": source_path,
            "heading_path": "",
            "doc_hash": doc_hash,
            "source_type": source_type,
            "title": title,
            "tags": "",
            "created_at_ts": created_at_ts,
            "chunk_index": idx,
            "total_chunks": total_chunks,
            "token_count": item["token_count"],
            "char_count": item["char_count"],
        }

        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                content=item["content"],
                source_path=source_path,
                heading_path="",
                score=0.0,
                metadata=chunk_metadata,
            )
        )

    return chunks


def parse_file(
    file_path: str | Path,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[RetrievedChunk]:
    """Parse any supported file format (.md, .pdf, .txt, .py, .json) into RetrievedChunks."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    source_type = EXTENSION_SOURCE_TYPE_MAP.get(suffix)

    if source_type is None:
        raise ValueError(
            f"Unsupported file format '{suffix}'. Supported: {list(EXTENSION_SOURCE_TYPE_MAP.keys())}"
        )

    fallback_ts = int(path.stat().st_ctime)
    norm_source_path = path.as_posix()

    if source_type == "markdown":
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="utf-8", errors="replace")
        return parse_markdown(
            content=content,
            source_path=norm_source_path,
            file_created_at_ts=fallback_ts,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    if source_type == "pdf":
        return parse_pdf(
            file_path=path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # txt, py, json
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="utf-8", errors="replace")

    if source_type == "json":
        # Format JSON for structured readability
        try:
            parsed_json = json.loads(content)
            content = json.dumps(parsed_json, indent=2)
        except Exception:
            pass

    return parse_text(
        content=content,
        source_path=norm_source_path,
        source_type=source_type,
        file_created_at_ts=fallback_ts,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def parse_file_to_document(
    file_path: str | Path,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> ParsedDocument:
    """Parse a file on disk and return a ParsedDocument model with metadata and chunks."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    source_type = EXTENSION_SOURCE_TYPE_MAP.get(suffix, "unknown")
    doc_hash = calculate_file_hash(path)
    chunks = parse_file(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    raw_content = ""
    frontmatter: dict[str, Any] = {}
    title = path.stem
    tags = ""
    created_at_ts = int(path.stat().st_ctime)

    if suffix in {".md", ".markdown"}:
        try:
            raw_content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw_content = path.read_text(encoding="utf-8", errors="replace")
        frontmatter, _ = extract_frontmatter(raw_content)
        tags = normalize_tags(frontmatter.get("tags"))
        if frontmatter.get("title"):
            title = str(frontmatter["title"])
        created_at_ts = parse_created_timestamp(frontmatter.get("created"), created_at_ts)
    elif suffix in {".txt", ".py", ".json"}:
        try:
            raw_content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw_content = path.read_text(encoding="utf-8", errors="replace")

    return ParsedDocument(
        source_path=path.as_posix(),
        doc_hash=doc_hash,
        source_type=source_type,
        raw_content=raw_content,
        title=title,
        tags=tags,
        created_at_ts=created_at_ts,
        frontmatter=frontmatter,
        chunks=chunks,
    )


class MarkdownASTParser:
    """Configurable document parser and two-stage chunking manager."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        headers_to_split_on: list[tuple[str, str]] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.headers_to_split_on = headers_to_split_on or DEFAULT_HEADERS_TO_SPLIT_ON

    def parse_markdown(self, content: str, source_path: str = "note.md") -> list[RetrievedChunk]:
        return parse_markdown(
            content=content,
            source_path=source_path,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def parse_file(self, file_path: str | Path) -> list[RetrievedChunk]:
        return parse_file(
            file_path=file_path,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def parse_file_to_document(self, file_path: str | Path) -> ParsedDocument:
        return parse_file_to_document(
            file_path=file_path,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
