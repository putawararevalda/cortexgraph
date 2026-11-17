"""Persistent ChromaDB vector store integration, OpenAI embeddings, and incremental sync.

Manages the `second_brain_notes` collection with cosine distance, SHA-256 doc_hash
incremental synchronization, orphaned chunk purging, and metadata-filtered retrieval.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import re
import socket
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.graph.state import RetrievedChunk
from src.rag.parser import calculate_file_hash, parse_file

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "second_brain_notes"
COSINE_METADATA: dict[str, Any] = {"hnsw:space": "cosine"}


class SyncStats(BaseModel):
    """Statistics returned by the incremental vault synchronization engine."""

    files_scanned: int = Field(default=0, description="Total files examined on disk")
    files_skipped: int = Field(default=0, description="Files unchanged (doc_hash match, 0 API calls)")
    chunks_upserted: int = Field(default=0, description="Chunks newly indexed or updated")
    chunks_deleted: int = Field(default=0, description="Orphaned or stale chunks purged from store")


_STOPWORDS: set[str] = {
    "a", "about", "after", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been",
    "before", "being", "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he",
    "her", "here", "hers", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "out", "over", "own", "same", "she", "so", "some", "such", "than",
    "that", "the", "their", "theirs", "them", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "you", "your",
    "list", "find", "retrieve", "explain", "describe", "show", "tell", "give", "overview",
    "file", "section"
}


def _deterministic_mock_embedding(text: str, dim: int = 1536) -> list[float]:
    """Generate a deterministic unit-length embedding vector from text for offline tests.

    Employs feature hashing of content tokens, n-grams, and sublinear weights to enable
    semantic term overlap and cosine similarity in offline zero-dependency mode.
    """
    clean_text = text.lower()
    words = [w for w in re.findall(r"[a-z0-9_\-]+", clean_text) if w not in _STOPWORDS and len(w) > 1]
    if not words:
        words = [w for w in re.findall(r"[a-z0-9_\-]+", clean_text) if len(w) > 1]

    if not words:
        seed_hash = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [
            math.sin(int.from_bytes(seed_hash[i % 32 : i % 32 + 2], "little") + i)
            for i in range(dim)
        ]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [round(x / norm, 6) for x in raw]

    vec = [0.0] * dim
    tokens = list(words)
    for i in range(len(words) - 1):
        tokens.append(f"{words[i]}_{words[i+1]}")

    for tok in tokens:
        h = int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:4], "little")
        idx = h % dim
        sign = 1.0 if (h & 0x8000) else -1.0
        weight = 1.0 + math.log(1.0 + len(tok))
        vec[idx] += sign * weight

    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [round(x / norm, 6) for x in vec]


class OpenAIEmbeddingFunctionWrapper:
    """Embedding function wrapping OpenAI text-embedding-3-small with call tracking and mock fallback."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "text-embedding-3-small",
        dimensions: int = 1536,
        offline: bool | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model_name = model_name
        self.dimensions = dimensions
        self.call_count = 0
        self.total_embeddings_generated = 0

        # Auto-detect offline mode if mock or empty key
        if offline is not None:
            self.offline = offline
        else:
            self.offline = (
                not self.api_key
                or self.api_key.startswith("sk-mock")
                or self.api_key.startswith("sk-test")
            )

        self._client: Any = None
        if not self.offline:
            try:
                import openai

                self._client = openai.OpenAI(api_key=self.api_key)
            except Exception as exc:
                logger.warning("Failed to initialize OpenAI client (%s). Using offline mock.", exc)
                self.offline = True

    def name(self) -> str:
        """ChromaDB embedding function identifier."""
        return "openai"

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self([query])[0]

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        """Generate embedding vectors for input texts."""
        texts = list(input)
        if not texts:
            return []

        self.call_count += 1
        self.total_embeddings_generated += len(texts)

        if self.offline or self._client is None:
            return [_deterministic_mock_embedding(t, dim=self.dimensions) for t in texts]

        try:
            response = self._client.embeddings.create(
                input=texts,
                model=self.model_name,
            )
            return [data.embedding for data in response.data]
        except Exception as exc:
            logger.warning("OpenAI embedding API call failed (%s). Falling back to mock vector.", exc)
            return [_deterministic_mock_embedding(t, dim=self.dimensions) for t in texts]


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Compute cosine distance between two float vectors."""
    if len(a) != len(b) or not a:
        return 1.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    similarity = dot / (norm_a * norm_b)
    # Cosine distance = 1 - similarity
    return max(0.0, min(2.0, 1.0 - similarity))


def _matches_where(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    """Evaluate a ChromaDB-style `where` query dictionary against a chunk metadata dict."""
    if not where:
        return True

    # Handle logical operators
    if "$and" in where:
        clauses = where["$and"]
        if isinstance(clauses, list):
            return all(_matches_where(metadata, clause) for clause in clauses)

    if "$or" in where:
        clauses = where["$or"]
        if isinstance(clauses, list):
            return any(_matches_where(metadata, clause) for clause in clauses)

    # Field-level conditions
    for key, condition in where.items():
        if key in {"$and", "$or"}:
            continue

        meta_val = metadata.get(key)

        # Exact match
        if not isinstance(condition, dict):
            if str(meta_val) != str(condition) and meta_val != condition:
                return False
            continue

        # Operator conditions
        for op, target in condition.items():
            if op == "$eq":
                if meta_val != target and str(meta_val) != str(target):
                    return False
            elif op == "$ne":
                if meta_val == target or str(meta_val) == str(target):
                    return False
            elif op == "$gt":
                if meta_val is None or meta_val <= target:
                    return False
            elif op == "$gte":
                if meta_val is None or meta_val < target:
                    return False
            elif op == "$lt":
                if meta_val is None or meta_val >= target:
                    return False
            elif op == "$lte":
                if meta_val is None or meta_val > target:
                    return False
            elif op == "$in":
                if meta_val not in target:
                    return False
            elif op == "$nin":
                if meta_val in target:
                    return False
            elif op == "$contains":
                # Substring match on string fields like tags
                str_val = str(meta_val or "")
                target_str = str(target or "")
                if target_str not in str_val:
                    return False

    return True


class InMemoryVectorCollection:
    """Robust in-memory vector collection adhering to the ChromaDB Collection API."""

    def __init__(
        self,
        name: str = DEFAULT_COLLECTION_NAME,
        metadata: dict[str, Any] | None = None,
        embedding_function: Any | None = None,
        persist_directory: str | None = None,
    ) -> None:
        self.name = name
        self.metadata = metadata or dict(COSINE_METADATA)
        self.embedding_function = embedding_function or OpenAIEmbeddingFunctionWrapper()
        self.persist_directory = persist_directory
        self._records: dict[str, dict[str, Any]] = {}
        self._load()

    @property
    def _storage_path(self) -> Path | None:
        if self.persist_directory:
            p = Path(self.persist_directory)
            p.mkdir(parents=True, exist_ok=True)
            return p / f"{self.name}_store.json"
        return None

    def _load(self) -> None:
        path = self._storage_path
        if path and path.exists():
            try:
                content = path.read_text(encoding="utf-8")
                data = json.loads(content)
                if isinstance(data, dict):
                    self._records = data
            except Exception as exc:
                logger.warning("Could not load persisted collection data from %s: %s", path, exc)

    def _save(self) -> None:
        path = self._storage_path
        if path:
            try:
                path.write_text(json.dumps(self._records), encoding="utf-8")
            except Exception as exc:
                logger.warning("Could not persist collection data to %s: %s", path, exc)

    def count(self) -> int:
        """Return the number of stored records."""
        return len(self._records)

    def get(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,  # noqa: ARG002
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """Retrieve stored records by ID or metadata filter."""
        selected_ids: list[str] = []
        selected_docs: list[str] = []
        selected_meta: list[dict[str, Any]] = []
        selected_embeddings: list[list[float]] = []

        target_ids = set(ids) if ids is not None else None

        for doc_id, record in list(self._records.items()):
            if target_ids is not None and doc_id not in target_ids:
                continue
            if where and not _matches_where(record.get("metadata", {}), where):
                continue

            selected_ids.append(doc_id)
            selected_docs.append(record.get("document", ""))
            selected_meta.append(dict(record.get("metadata", {})))
            selected_embeddings.append(record.get("embedding", []))

        start = offset or 0
        end = start + limit if limit is not None else len(selected_ids)

        return {
            "ids": selected_ids[start:end],
            "documents": selected_docs[start:end],
            "metadatas": selected_meta[start:end],
            "embeddings": selected_embeddings[start:end],
        }

    def upsert(
        self,
        ids: list[str],
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        embeddings: list[list[float]] | None = None,
        uris: list[str] | None = None,  # noqa: ARG002
    ) -> None:
        """Upsert records into the collection."""
        docs = documents or ["" for _ in ids]
        metas = metadatas or [{} for _ in ids]

        if embeddings is None:
            embeddings = self.embedding_function(docs)

        for doc_id, doc, meta, emb in zip(ids, docs, metas, embeddings, strict=False):
            self._records[doc_id] = {
                "id": doc_id,
                "document": doc,
                "metadata": dict(meta),
                "embedding": list(emb),
            }
        self._save()

    def add(
        self,
        ids: list[str],
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        embeddings: list[list[float]] | None = None,
        uris: list[str] | None = None,
    ) -> None:
        """Alias to upsert for API parity."""
        self.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings, uris=uris)

    def delete(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> None:
        """Delete records matching IDs or metadata filter."""
        to_delete: set[str] = set()

        if ids is not None:
            to_delete.update(ids)

        if where is not None:
            for doc_id, record in self._records.items():
                if _matches_where(record.get("metadata", {}), where):
                    to_delete.add(doc_id)

        for doc_id in to_delete:
            self._records.pop(doc_id, None)

        if to_delete:
            self._save()

    def query(
        self,
        query_texts: list[str] | None = None,
        query_embeddings: list[list[float]] | None = None,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        """Perform cosine similarity vector search over the collection."""
        if not self._records:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        # Resolve query embedding
        if query_embeddings is not None and len(query_embeddings) > 0:
            target_emb = query_embeddings[0]
        elif query_texts is not None and len(query_texts) > 0:
            target_emb = self.embedding_function(query_texts)[0]
        else:
            raise ValueError("Must provide either query_texts or query_embeddings")

        candidates: list[tuple[float, str, str, dict[str, Any]]] = []

        for doc_id, record in self._records.items():
            meta = record.get("metadata", {})
            if where and not _matches_where(meta, where):
                continue

            emb = record.get("embedding", [])
            dist = _cosine_distance(target_emb, emb)
            candidates.append((dist, doc_id, record.get("document", ""), meta))

        # Sort by distance ascending (lower distance = higher similarity)
        candidates.sort(key=lambda x: x[0])
        top_candidates = candidates[:n_results]

        return {
            "ids": [[c[1] for c in top_candidates]],
            "documents": [[c[2] for c in top_candidates]],
            "metadatas": [[c[3] for c in top_candidates]],
            "distances": [[c[0] for c in top_candidates]],
        }


class InMemoryChromaClient:
    """Drop-in Chroma client managing InMemoryVectorCollection instances."""

    def __init__(self, persist_directory: str | None = None) -> None:
        self.persist_directory = persist_directory
        self._collections: dict[str, InMemoryVectorCollection] = {}

    def get_or_create_collection(
        self,
        name: str = DEFAULT_COLLECTION_NAME,
        metadata: dict[str, Any] | None = None,
        embedding_function: Any | None = None,
    ) -> InMemoryVectorCollection:
        if name not in self._collections:
            self._collections[name] = InMemoryVectorCollection(
                name=name,
                metadata=metadata,
                embedding_function=embedding_function,
                persist_directory=self.persist_directory,
            )
        return self._collections[name]

    def create_collection(
        self,
        name: str = DEFAULT_COLLECTION_NAME,
        metadata: dict[str, Any] | None = None,
        embedding_function: Any | None = None,
    ) -> InMemoryVectorCollection:
        return self.get_or_create_collection(name, metadata=metadata, embedding_function=embedding_function)

    def get_collection(self, name: str) -> InMemoryVectorCollection:
        if name not in self._collections:
            raise KeyError(f"Collection '{name}' does not exist.")
        return self._collections[name]

    def delete_collection(self, name: str) -> None:
        self._collections.pop(name, None)

    def reset(self) -> None:
        self._collections.clear()

    def heartbeat(self) -> int:
        return 1


def is_chroma_server_ready(host: str, port: int, timeout: float = 0.2) -> bool:
    """Test TCP socket reachability for an external ChromaDB HTTP server."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def get_chroma_client(
    settings: Settings | None = None,
    persist_directory: str | None = None,
    in_memory: bool = False,
) -> Any:
    """ChromaDB client factory supporting HTTP server, local persistence, and in-memory modes."""
    app_settings = settings or get_settings()
    target_dir = persist_directory or "./data/chroma_db"

    # 1. Connect via HTTP if remote server is running
    if (
        not in_memory
        and app_settings.chroma_host
        and is_chroma_server_ready(app_settings.chroma_host, app_settings.chroma_port)
    ):
        try:
            import chromadb

            client = chromadb.HttpClient(
                host=app_settings.chroma_host,
                port=app_settings.chroma_port,
            )
            client.heartbeat()
            logger.info(
                "Connected to remote ChromaDB at %s:%s",
                app_settings.chroma_host,
                app_settings.chroma_port,
            )
            return client
        except Exception as exc:
            logger.warning("Remote ChromaDB connection failed: %s", exc)

    # 2. Local mode: Persistent or in-memory
    if in_memory:
        return InMemoryChromaClient(persist_directory=None)

    # Fallback to persistent InMemoryChromaClient to avoid native DLL issues on Windows
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    return InMemoryChromaClient(persist_directory=target_dir)


def build_chroma_where_clause(metadata_filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert a high-level metadata filters dict into ChromaDB `$and` / operator clauses."""
    if not metadata_filters:
        return None

    clauses: list[dict[str, Any]] = []

    # 1. Source type filter
    source_type = metadata_filters.get("source_type") or metadata_filters.get("source_type_filter")
    if source_type:
        clauses.append({"source_type": {"$eq": str(source_type).lower()}})

    # 2. Temporal filters (Unix timestamp)
    created_after = metadata_filters.get("created_after_ts") or metadata_filters.get("created_after")
    if created_after is not None:
        with contextlib.suppress(ValueError, TypeError):
            clauses.append({"created_at_ts": {"$gte": int(created_after)}})

    created_before = metadata_filters.get("created_before_ts") or metadata_filters.get("created_before")
    if created_before is not None:
        with contextlib.suppress(ValueError, TypeError):
            clauses.append({"created_at_ts": {"$lte": int(created_before)}})

    # 3. Tag filters
    tags = metadata_filters.get("required_tags") or metadata_filters.get("tags") or metadata_filters.get("tag")
    if tags:
        tag_list: list[str] = []
        if isinstance(tags, (list, tuple, set)):
            tag_list = [str(t).strip().lower().lstrip("#") for t in tags if str(t).strip()]
        elif isinstance(tags, str):
            clean = tags.replace(",", " ").split()
            tag_list = [t.strip().lower().lstrip("#") for t in clean if t.strip()]

        for tag in tag_list:
            # Normalized tags are formatted like `,ai,agents,`
            clauses.append({"tags": {"$contains": f",{tag},"}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


class VectorStore:
    """ChromaDB Vector Store manager for CortexGraph Second Brain notes."""

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        client: Any | None = None,
        embedding_function: Any | None = None,
        settings: Settings | None = None,
        in_memory: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self.collection_name = collection_name
        self.embedding_function = embedding_function or OpenAIEmbeddingFunctionWrapper(
            api_key=self.settings.openai_api_key,
            model_name=self.settings.openai_embedding_model,
        )

        if client is not None:
            self.client = client
        else:
            self.client = get_chroma_client(settings=self.settings, in_memory=in_memory)

        # Initialize or get the persistent collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata=COSINE_METADATA,
                embedding_function=self.embedding_function,
            )
        except Exception as exc:
            logger.warning(
                "Failed to get_or_create_collection from client (%s). Using InMemoryVectorCollection.",
                exc,
            )
            self.collection = InMemoryVectorCollection(
                name=self.collection_name,
                metadata=COSINE_METADATA,
                embedding_function=self.embedding_function,
                persist_directory="./data/chroma_db",
            )

    def sync_vault(
        self,
        vault_path: str | Path = "data/sample_vault",
        force_reindex: bool = False,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> SyncStats:
        """Incrementally synchronize a local document vault into ChromaDB using SHA-256 doc_hash.

        Rules:
        1. Compare stored `doc_hash` in ChromaDB with current file `doc_hash`.
        2. If hashes match, skip the file completely (0 embedding API calls).
        3. If hash differs, purge existing chunks and upsert new chunks with IDs `f"{source_path}::chunk_{idx}"`.
        4. Detect deleted files on disk and purge their orphaned chunks from ChromaDB.
        """
        vault_dir = Path(vault_path)
        if not vault_dir.exists():
            raise FileNotFoundError(f"Vault directory not found: {vault_path}")

        stats = SyncStats()
        supported_extensions = {".md", ".markdown", ".pdf", ".txt", ".py", ".json"}

        # 1. Discover all candidate notes on disk
        discovered_files: list[Path] = []
        for p in vault_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in supported_extensions:
                discovered_files.append(p)

        discovered_files.sort(key=lambda p: str(p.as_posix()))
        stats.files_scanned = len(discovered_files)

        active_source_paths: set[str] = set()

        # 2. Process each file incrementally
        for file_path in discovered_files:
            try:
                norm_source_path = file_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
            except ValueError:
                norm_source_path = file_path.as_posix()

            active_source_paths.add(norm_source_path)
            current_doc_hash = calculate_file_hash(file_path)

            # Query existing chunks by source_path
            existing = self.collection.get(
                where={"source_path": norm_source_path},
                include=["metadatas"],
            )
            existing_ids: list[str] = existing.get("ids", [])
            existing_metas: list[dict[str, Any]] = existing.get("metadatas", [])

            # Check for unchanged document hash
            if not force_reindex and existing_ids and existing_metas:
                stored_hash = existing_metas[0].get("doc_hash", "")
                if stored_hash == current_doc_hash:
                    # Idempotent match: Skip embedding completely (0 OpenAI calls)
                    stats.files_skipped += 1
                    continue

                # File has changed: Delete stale chunks before upserting new ones
                self.collection.delete(ids=existing_ids)
                stats.chunks_deleted += len(existing_ids)

            # Parse new chunks
            try:
                chunks = parse_file(
                    file_path=file_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            except Exception as parse_exc:
                logger.warning("Failed parsing %s: %s", file_path, parse_exc)
                continue

            if not chunks:
                continue

            # Prepare deterministic IDs and metadata
            ids: list[str] = []
            docs: list[str] = []
            metas: list[dict[str, Any]] = []

            for idx, chunk in enumerate(chunks):
                chunk_id = f"{norm_source_path}::chunk_{idx}"
                ids.append(chunk_id)
                docs.append(chunk.content)

                # Ensure metadata contains required fields and primitive values
                meta = dict(chunk.metadata)
                meta["source_path"] = norm_source_path
                meta["heading_path"] = chunk.heading_path
                meta["doc_hash"] = current_doc_hash
                meta["chunk_index"] = idx
                meta["total_chunks"] = len(chunks)

                # Sanitize metadata values to Chroma-safe types
                sanitized_meta: dict[str, str | int | float | bool] = {}
                for k, v in meta.items():
                    if isinstance(v, (str, int, float, bool)):
                        sanitized_meta[k] = v
                    elif v is None:
                        sanitized_meta[k] = ""
                    else:
                        sanitized_meta[k] = str(v)

                metas.append(sanitized_meta)

            self.collection.upsert(ids=ids, documents=docs, metadatas=metas)
            stats.chunks_upserted += len(ids)

        # 3. Detect and purge orphaned chunks for deleted files
        try:
            vault_prefix = (
                vault_dir.resolve().relative_to(Path.cwd().resolve()).as_posix()
                if vault_dir.is_absolute() or str(vault_dir).startswith(".")
                else vault_dir.as_posix()
            )
        except Exception:
            vault_prefix = vault_dir.as_posix()

        all_records = self.collection.get(include=["metadatas"])
        all_ids = all_records.get("ids", [])
        all_metas = all_records.get("metadatas", [])

        orphaned_ids: list[str] = []
        for rec_id, rec_meta in zip(all_ids, all_metas, strict=False):
            rec_source = rec_meta.get("source_path", "")
            if not rec_source:
                continue

            # Check if this chunk belongs to the target vault
            if (
                (rec_source.startswith(vault_prefix) or vault_prefix in rec_source)
                and rec_source not in active_source_paths
                and not Path(rec_source).exists()
            ):
                orphaned_ids.append(rec_id)

        if orphaned_ids:
            self.collection.delete(ids=orphaned_ids)
            stats.chunks_deleted += len(orphaned_ids)

        logger.info(
            "Vault sync complete: scanned=%d, skipped=%d, upserted=%d, deleted=%d",
            stats.files_scanned,
            stats.files_skipped,
            stats.chunks_upserted,
            stats.chunks_deleted,
        )
        return stats

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Perform similarity search over the vector store."""
        return self.retrieve_chunks(query=query, top_k=k, metadata_filters=filter)

    def retrieve_chunks(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Perform metadata-filtered similarity search with automatic relaxation fallback.

        If strict metadata filters return 0 results on a rewrite pass, falls back
        to unfiltered similarity search. Diversifies chunks across source documents.
        """
        where_clause = build_chroma_where_clause(metadata_filters)
        fetch_k = max(top_k * 3, 12)

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=fetch_k,
                where=where_clause,
            )
        except Exception as exc:
            logger.warning("Metadata-filtered query failed (%s). Retrying without where clause.", exc)
            results = self.collection.query(
                query_texts=[query],
                n_results=fetch_k,
                where=None,
            )

        # Automatic relaxation fallback: if filtered search produced 0 results, retry unfiltered
        ids_list = results.get("ids", [[]])[0]
        if not ids_list and where_clause is not None:
            logger.info("Filtered query returned 0 chunks. Executing relaxation fallback query.")
            results = self.collection.query(
                query_texts=[query],
                n_results=fetch_k,
                where=None,
            )
            ids_list = results.get("ids", [[]])[0]

        if not ids_list:
            return []

        docs_list = results.get("documents", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0] if "distances" in results else [0.0] * len(ids_list)

        retrieved: list[RetrievedChunk] = []
        for i, chunk_id in enumerate(ids_list):
            doc = docs_list[i] if i < len(docs_list) else ""
            meta = metas_list[i] if i < len(metas_list) else {}
            dist = distances[i] if i < len(distances) else 0.0

            # Cosine distance d in [0, 2] -> score in [0.0, 1.0]
            similarity_score = max(0.0, min(1.0, 1.0 - float(dist)))

            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    content=doc,
                    source_path=meta.get("source_path", ""),
                    heading_path=meta.get("heading_path", ""),
                    score=round(similarity_score, 4),
                    metadata=dict(meta),
                )
            )

        # Source diversification: ensure distinct document representation across top_k
        diverse_retrieved: list[RetrievedChunk] = []
        seen_sources: set[str] = set()

        for chunk in retrieved:
            src = chunk.source_path
            if src not in seen_sources:
                seen_sources.add(src)
                diverse_retrieved.append(chunk)
            if len(diverse_retrieved) >= top_k:
                break

        if len(diverse_retrieved) < top_k:
            for chunk in retrieved:
                if chunk not in diverse_retrieved:
                    diverse_retrieved.append(chunk)
                if len(diverse_retrieved) >= top_k:
                    break

        return diverse_retrieved


# Global default instance
_default_vectorstore: VectorStore | None = None


def get_vectorstore(
    settings: Settings | None = None,
    in_memory: bool = False,
    force_new: bool = False,
) -> VectorStore:
    """Retrieve or initialize the global VectorStore instance."""
    global _default_vectorstore
    if _default_vectorstore is None or force_new:
        _default_vectorstore = VectorStore(
            collection_name=DEFAULT_COLLECTION_NAME,
            settings=settings,
            in_memory=in_memory,
        )
    return _default_vectorstore


def retrieve_chunks(
    query: str,
    top_k: int = 5,
    metadata_filters: dict[str, Any] | None = None,
    vectorstore: VectorStore | None = None,
) -> list[RetrievedChunk]:
    """Retrieve knowledge chunks for the LangGraph CRAG retriever node."""
    store = vectorstore or get_vectorstore()
    return store.retrieve_chunks(query=query, top_k=top_k, metadata_filters=metadata_filters)


def sync_vault(
    vault_path: str | Path = "data/sample_vault",
    force_reindex: bool = False,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    vectorstore: VectorStore | None = None,
) -> SyncStats:
    """Trigger vault synchronization and return sync statistics."""
    store = vectorstore or get_vectorstore()
    return store.sync_vault(
        vault_path=vault_path,
        force_reindex=force_reindex,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
