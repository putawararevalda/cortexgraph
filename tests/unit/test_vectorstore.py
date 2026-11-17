"""Unit tests for ChromaDB vector store, SHA-256 incremental sync, and ingest API."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from src.graph.state import RetrievedChunk
from src.rag.vectorstore import (
    DEFAULT_COLLECTION_NAME,
    OpenAIEmbeddingFunctionWrapper,
    VectorStore,
)


class TestVectorStoreInitialization:
    """Verifies collection configuration, cosine metric, and embedding function wrapper."""

    def test_embedding_function_call_tracking(self) -> None:
        """Verify wrapper tracks call count and generates deterministic vectors offline."""
        wrapper = OpenAIEmbeddingFunctionWrapper(api_key="sk-mock-offline-key", dimensions=64)
        assert wrapper.name() == "openai"
        assert wrapper.call_count == 0

        vectors = wrapper(["first note text", "second note text"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 64
        assert wrapper.call_count == 1
        assert wrapper.total_embeddings_generated == 2

        # Verify embed_query alias
        q_vec = wrapper.embed_query("search query")
        assert len(q_vec) == 64
        assert wrapper.call_count == 2

    def test_vectorstore_default_collection_configuration(self) -> None:
        """Verify second_brain_notes collection is configured with cosine distance."""
        store = VectorStore(
            collection_name=DEFAULT_COLLECTION_NAME,
            in_memory=True,
        )
        assert store.collection_name == "second_brain_notes"
        assert store.collection.metadata.get("hnsw:space") == "cosine"


class TestIncrementalVaultSync:
    """Verifies SHA-256 doc_hash diffing, zero-duplicate sync, and orphan pruning."""

    def test_sync_vault_idempotency_and_zero_calls(self, tmp_path: Path) -> None:
        """Consecutive sync runs verify files_skipped == total_files and 0 API calls."""
        # Setup temporary mock vault with 3 notes
        vault_dir = tmp_path / "test_vault"
        vault_dir.mkdir(parents=True)

        for i in range(1, 4):
            note = vault_dir / f"note_{i}.md"
            note.write_text(
                f"---\ntitle: Note {i}\ntags: [ai, testing]\ncreated: 2026-10-15\n---\n\n"
                f"# Note {i}\n\nContent for note {i} detailing architecture.\n",
                encoding="utf-8",
            )

        mock_ef = OpenAIEmbeddingFunctionWrapper(api_key="sk-mock-key", dimensions=32)
        store = VectorStore(
            collection_name="test_sync_idempotency",
            embedding_function=mock_ef,
            in_memory=True,
        )

        # Run 1: Initial Ingest
        stats_1 = store.sync_vault(vault_path=vault_dir)
        assert stats_1.files_scanned == 3
        assert stats_1.files_skipped == 0
        assert stats_1.chunks_upserted >= 3
        assert stats_1.chunks_deleted == 0
        assert store.collection.count() >= 3

        calls_after_run1 = mock_ef.call_count
        assert calls_after_run1 > 0

        # Run 2: Immediate duplicate run without disk changes
        stats_2 = store.sync_vault(vault_path=vault_dir)
        assert stats_2.files_scanned == 3
        assert stats_2.files_skipped == 3
        assert stats_2.chunks_upserted == 0
        assert stats_2.chunks_deleted == 0

        # 0 new OpenAI embedding calls on duplicate run
        assert mock_ef.call_count == calls_after_run1

    def test_updating_note_updates_only_that_note_chunks(self, tmp_path: Path) -> None:
        """Modifying one note alters only its chunks in ChromaDB."""
        vault_dir = tmp_path / "update_vault"
        vault_dir.mkdir(parents=True)

        note_a = vault_dir / "note_a.md"
        note_b = vault_dir / "note_b.md"
        note_a.write_text("# Note A\nOriginal content A.\n", encoding="utf-8")
        note_b.write_text("# Note B\nOriginal content B.\n", encoding="utf-8")

        store = VectorStore(
            collection_name="test_update_note",
            in_memory=True,
        )

        stats_1 = store.sync_vault(vault_path=vault_dir)
        assert stats_1.files_scanned == 2
        assert stats_1.chunks_upserted >= 2

        # Update ONLY note_a
        note_a.write_text("# Note A\nCompletely modified and updated content A.\n", encoding="utf-8")

        stats_2 = store.sync_vault(vault_path=vault_dir)
        assert stats_2.files_scanned == 2
        assert stats_2.files_skipped == 1  # note_b skipped
        assert stats_2.chunks_upserted >= 1  # note_a re-upserted
        assert stats_2.chunks_deleted >= 1  # old note_a chunk removed

    def test_deleting_note_removes_chunks_cleanly(self, tmp_path: Path) -> None:
        """Deleting a note from disk purges its orphaned chunks from ChromaDB."""
        vault_dir = tmp_path / "orphan_vault"
        vault_dir.mkdir(parents=True)

        note_1 = vault_dir / "note_1.md"
        note_2 = vault_dir / "note_2.md"
        note_1.write_text("# Note 1\nKept note.\n", encoding="utf-8")
        note_2.write_text("# Note 2\nTo be deleted note.\n", encoding="utf-8")

        store = VectorStore(
            collection_name="test_orphan_purge",
            in_memory=True,
        )

        stats_1 = store.sync_vault(vault_path=vault_dir)
        assert stats_1.files_scanned == 2
        initial_chunk_count = store.collection.count()
        assert initial_chunk_count >= 2

        # Remove note_2 from disk
        note_2.unlink()
        assert not note_2.exists()

        # Sync again
        stats_2 = store.sync_vault(vault_path=vault_dir)
        assert stats_2.files_scanned == 1
        assert stats_2.files_skipped == 1
        assert stats_2.chunks_deleted >= 1
        assert store.collection.count() < initial_chunk_count

    def test_deterministic_chunk_ids_format(self, tmp_path: Path) -> None:
        """Verify IDs adhere to `f"{source_path}::chunk_{idx}"` format."""
        vault_dir = tmp_path / "id_vault"
        vault_dir.mkdir(parents=True)

        note = vault_dir / "syntax_note.md"
        note.write_text("# Title\nBody chunk 1.\n## Sub\nBody chunk 2.\n", encoding="utf-8")

        store = VectorStore(collection_name="test_id_format", in_memory=True)
        store.sync_vault(vault_path=vault_dir)

        all_records = store.collection.get()
        assert len(all_records["ids"]) >= 1
        for chunk_id in all_records["ids"]:
            assert "::chunk_" in chunk_id


class TestMetadataFilteredRetrieval:
    """Verifies ChromaDB where filtering and automatic relaxation fallback."""

    @pytest.fixture
    def populated_store(self) -> VectorStore:
        store = VectorStore(collection_name="test_filtering", in_memory=True)
        store.collection.upsert(
            ids=["chunk_1", "chunk_2", "chunk_3"],
            documents=[
                "LangGraph state persistence with SQLite checkpointers.",
                "ChromaDB cosine vector search and metadata filters.",
                "Docker compose configuration for FastAPI and Chroma.",
            ],
            metadatas=[
                {
                    "source_path": "notes/langgraph.md",
                    "source_type": "markdown",
                    "tags": ",langgraph,agents,",
                    "created_at_ts": 1000,
                },
                {
                    "source_path": "notes/chromadb.md",
                    "source_type": "markdown",
                    "tags": ",vector,database,chroma,",
                    "created_at_ts": 2000,
                },
                {
                    "source_path": "infra/docker.txt",
                    "source_type": "text",
                    "tags": ",docker,devops,",
                    "created_at_ts": 3000,
                },
            ],
        )
        return store

    def test_filter_by_source_type(self, populated_store: VectorStore) -> None:
        chunks = populated_store.retrieve_chunks(
            query="Docker setup",
            top_k=5,
            metadata_filters={"source_type": "text"},
        )
        assert len(chunks) == 1
        assert chunks[0].source_path == "infra/docker.txt"

    def test_filter_by_tags(self, populated_store: VectorStore) -> None:
        chunks = populated_store.retrieve_chunks(
            query="vector search",
            top_k=5,
            metadata_filters={"required_tags": ["chroma"]},
        )
        assert len(chunks) == 1
        assert chunks[0].source_path == "notes/chromadb.md"

    def test_filter_by_temporal_range(self, populated_store: VectorStore) -> None:
        chunks = populated_store.retrieve_chunks(
            query="state and vector",
            top_k=5,
            metadata_filters={"created_after_ts": 1500},
        )
        # Should match chunk 2 (2000) and chunk 3 (3000), but not chunk 1 (1000)
        source_paths = [c.source_path for c in chunks]
        assert "notes/langgraph.md" not in source_paths
        assert "notes/chromadb.md" in source_paths

    def test_automatic_relaxation_fallback_on_zero_results(
        self, populated_store: VectorStore
    ) -> None:
        """Strict metadata filter returning 0 chunks falls back to unfiltered similarity search."""
        # Filter on created_after_ts far in future yields 0 matches strictly
        chunks = populated_store.retrieve_chunks(
            query="LangGraph checkpointer",
            top_k=2,
            metadata_filters={"created_after_ts": 9999999999},
        )
        # Fallback triggers and returns top chunks
        assert len(chunks) > 0
        assert isinstance(chunks[0], RetrievedChunk)


class TestSampleVaultFullIngestAndAPI:
    """Verifies DoD on data/sample_vault/ and FastAPI POST /api/v1/ingest."""

    @pytest.mark.asyncio
    async def test_post_api_v1_ingest_consecutive_runs(
        self, async_client: httpx.AsyncClient
    ) -> None:
        """Running POST /api/v1/ingest twice on data/sample_vault verifies files_skipped == 25."""
        sample_vault_path = "data/sample_vault"

        # Run 1: Index vault (force_reindex ensures clean baseline regardless of prior run artifacts)
        resp_1 = await async_client.post(
            "/api/v1/ingest", json={"vault_path": sample_vault_path, "force_reindex": True}
        )
        assert resp_1.status_code == 200
        data_1 = resp_1.json()
        assert data_1["status"] == "success"
        assert data_1["files_scanned"] == 25
        assert data_1["chunks_upserted"] > 0

        # Run 2: Consecutive call immediately after without force_reindex
        resp_2 = await async_client.post(
            "/api/v1/ingest", json={"vault_path": sample_vault_path, "force_reindex": False}
        )
        assert resp_2.status_code == 200
        data_2 = resp_2.json()
        assert data_2["status"] == "success"
        assert data_2["files_scanned"] == 25
        assert data_2["files_skipped"] == 25  # All 25 notes skipped idempotently
        assert data_2["chunks_upserted"] == 0
        assert data_2["chunks_deleted"] == 0

    @pytest.mark.asyncio
    async def test_post_api_v1_ingest_missing_path(
        self, async_client: httpx.AsyncClient
    ) -> None:
        """Verify 404 is returned for non-existent vault path."""
        resp = await async_client.post("/api/v1/ingest", json={"vault_path": "invalid/vault/dir"})
        assert resp.status_code == 404
