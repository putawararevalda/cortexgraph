"""FastAPI endpoint and CLI for SHA-256 incremental vault ingestion into ChromaDB."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.rag.vectorstore import SyncStats, get_vectorstore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Ingest"])


class IngestRequest(BaseModel):
    """Payload for triggering vault synchronization."""

    vault_path: str = Field(
        default="data/sample_vault",
        description="Path to the Markdown/PDF vault directory on disk",
    )
    force_reindex: bool = Field(
        default=False,
        description="If True, reindexes all files regardless of doc_hash match",
    )
    chunk_size: int = Field(
        default=512,
        ge=64,
        le=2048,
        description="Maximum tokens per chunk",
    )
    chunk_overlap: int = Field(
        default=64,
        ge=0,
        le=512,
        description="Token overlap between adjacent chunks",
    )


class IngestResponse(BaseModel):
    """Summary statistics returned upon vault sync completion."""

    status: str = Field(default="success", description="Status code string")
    vault_path: str = Field(..., description="Canonical path of the ingested vault")
    files_scanned: int = Field(..., description="Total files examined on disk")
    files_skipped: int = Field(..., description="Files skipped due to matching doc_hash (0 API calls)")
    chunks_upserted: int = Field(..., description="Total chunks newly inserted or updated")
    chunks_deleted: int = Field(..., description="Total orphaned or stale chunks removed")
    duration_seconds: float = Field(..., description="Total ingestion execution time in seconds")


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger incremental vault ingestion",
    description="Synchronizes a local Markdown vault into ChromaDB using SHA-256 doc_hash diffing.",
)
async def ingest_vault(
    payload: Annotated[IngestRequest | None, Body()] = None,
    settings: Annotated[Settings, Depends(get_settings)] = None,  # type: ignore[assignment]
) -> IngestResponse:
    """Execute incremental vault synchronization."""
    request = payload if payload is not None else IngestRequest()
    if settings is None:
        settings = get_settings()

    start_time = time.perf_counter()
    vault_path = Path(request.vault_path)
    if not vault_path.exists():
        repo_root = Path(__file__).resolve().parent.parent.parent
        candidate = repo_root / request.vault_path
        if candidate.exists():
            vault_path = candidate
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Vault path '{request.vault_path}' does not exist on disk.",
            )

    try:
        vectorstore = get_vectorstore(settings=settings)
        stats: SyncStats = vectorstore.sync_vault(
            vault_path=vault_path,
            force_reindex=request.force_reindex,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
    except Exception as exc:
        logger.error("Failed during vault ingestion: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        ) from exc

    duration = round(time.perf_counter() - start_time, 4)

    return IngestResponse(
        status="success",
        vault_path=str(vault_path.as_posix()),
        files_scanned=stats.files_scanned,
        files_skipped=stats.files_skipped,
        chunks_upserted=stats.chunks_upserted,
        chunks_deleted=stats.chunks_deleted,
        duration_seconds=duration,
    )


def run_cli_ingest(args: list[str] | None = None) -> int:
    """CLI runner for triggering vault ingestion from the terminal."""
    parser = argparse.ArgumentParser(
        description="CortexGraph Vault Ingestion CLI - SHA-256 Incremental ChromaDB Sync"
    )
    parser.add_argument(
        "--vault-path",
        default="data/sample_vault",
        help="Path to the personal notes vault directory (default: data/sample_vault)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing of all files, bypassing SHA-256 doc_hash check",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Maximum chunk size in tokens (default: 512)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        help="Overlap between adjacent chunks in tokens (default: 64)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output sync results as JSON",
    )

    parsed = parser.parse_args(args)
    settings = get_settings()

    vault_dir = Path(parsed.vault_path)
    if not vault_dir.exists():
        repo_root = Path(__file__).resolve().parent.parent.parent
        candidate = repo_root / parsed.vault_path
        if candidate.exists():
            vault_dir = candidate
        else:
            print(f"Error: Vault path '{parsed.vault_path}' does not exist.", file=sys.stderr)
            return 1

    print(f"Starting vault sync: {vault_dir} (force={parsed.force}) ...")
    start = time.perf_counter()

    try:
        store = get_vectorstore(settings=settings)
        stats = store.sync_vault(
            vault_path=vault_dir,
            force_reindex=parsed.force,
            chunk_size=parsed.chunk_size,
            chunk_overlap=parsed.chunk_overlap,
        )
    except Exception as exc:
        print(f"Fatal error during ingestion: {exc}", file=sys.stderr)
        return 1

    duration = round(time.perf_counter() - start, 4)
    result_dict: dict[str, Any] = {
        "status": "success",
        "vault_path": str(vault_dir.as_posix()),
        "files_scanned": stats.files_scanned,
        "files_skipped": stats.files_skipped,
        "chunks_upserted": stats.chunks_upserted,
        "chunks_deleted": stats.chunks_deleted,
        "duration_seconds": duration,
    }

    if parsed.json:
        print(json.dumps(result_dict, indent=2))
    else:
        print("Vault ingestion complete:")
        print(f"  Files Scanned:   {stats.files_scanned}")
        print(f"  Files Skipped:   {stats.files_skipped} (idempotent 0-call matches)")
        print(f"  Chunks Upserted: {stats.chunks_upserted}")
        print(f"  Chunks Deleted:  {stats.chunks_deleted}")
        print(f"  Elapsed Time:    {duration}s")

    return 0


if __name__ == "__main__":
    sys.exit(run_cli_ingest())
