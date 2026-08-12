"""Vector-store helpers for RAG retrieval.

This module is reserved for ChromaDB-backed storage and similarity search. The
current functions are placeholders so the repository has a clean integration
point before the full retrieval pipeline is built.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned from the vector store."""

    chunk_id: str
    source_path: str
    text: str
    score: float


def index_chunks() -> None:
    """Placeholder for future ChromaDB indexing logic."""


def search_chunks(query: str, limit: int = 5) -> list[RetrievedChunk]:
    """Placeholder for future ChromaDB similarity search."""

    return []

