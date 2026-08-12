"""Document ingestion helpers for the local knowledge base.

This module will eventually load files from ``knowledge_base/`` and split them
into retrieval-friendly chunks. The implementation is intentionally minimal for
now so the RAG pipeline can be wired in incrementally.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KnowledgeChunk:
    """A single chunk produced from a knowledge-base document."""

    source_path: str
    chunk_id: str
    text: str


def load_documents(base_path: Path) -> list[Path]:
    """Return candidate knowledge-base document paths.

    Placeholder only; full file loading and filtering logic will be added when
    the ingestion pipeline is implemented.
    """

    return [path for path in base_path.rglob("*") if path.is_file()]


def chunk_documents(document_paths: list[Path]) -> list[KnowledgeChunk]:
    """Convert raw document paths into placeholder chunks.

    The real implementation will parse file contents and split them into
    semantically useful chunks.
    """

    return [
        KnowledgeChunk(source_path=str(path), chunk_id=f"{index}", text="")
        for index, path in enumerate(document_paths)
    ]

