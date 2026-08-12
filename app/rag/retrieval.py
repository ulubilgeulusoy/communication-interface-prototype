"""ChromaDB-backed storage and similarity search for RAG chunks.

This module is responsible only for vector-store operations:

- creating or opening a persistent local Chroma collection
- upserting chunk embeddings with metadata
- querying the vector store for nearest-neighbor matches

It intentionally does not generate embeddings or prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from .embeddings import EmbeddingBatch, EmbeddingItem
from .ingest import KnowledgeChunk


DEFAULT_COLLECTION_NAME = "knowledge_chunks"
DEFAULT_TOP_K = 5
DEFAULT_VECTOR_STORE_PATH = Path(__file__).resolve().parent.parent.parent / "vector_store"


class RetrievalServiceError(Exception):
    """Raised when vector-store indexing or retrieval fails."""


@dataclass(frozen=True)
class IndexedChunk:
    """Chunk data paired with its embedding for vector-store indexing."""

    chunk: KnowledgeChunk
    vector: list[float]


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned from similarity search."""

    chunk_id: str
    text: str
    source_path: str
    filename: str
    category: str
    chunk_index: int
    start_char: int
    end_char: int
    similarity_distance: float | None
    collection_name: str


class ChromaRetrievalStore:
    """Persistent ChromaDB wrapper for chunk indexing and search."""

    def __init__(
        self,
        *,
        store_path: Path = DEFAULT_VECTOR_STORE_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self.store_path = Path(store_path)
        self.collection_name = collection_name
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.store_path))

    def upsert_chunks(
        self,
        chunks: list[KnowledgeChunk],
        embeddings: EmbeddingBatch,
        *,
        collection_name: str | None = None,
    ) -> None:
        """Insert or update chunk embeddings and metadata in Chroma."""

        if len(chunks) != len(embeddings.items):
            raise RetrievalServiceError("Chunk count must match embedding count.")

        collection = self._get_collection(collection_name)
        indexed_chunks = self._pair_chunks_with_embeddings(chunks, embeddings)

        try:
            collection.upsert(
                ids=[item.chunk.chunk_id for item in indexed_chunks],
                embeddings=[item.vector for item in indexed_chunks],
                documents=[item.chunk.text for item in indexed_chunks],
                metadatas=[self._metadata_for_chunk(item.chunk) for item in indexed_chunks],
            )
        except Exception as exc:  # pragma: no cover - Chroma raises library-specific errors
            raise RetrievalServiceError("Failed to upsert chunks into ChromaDB.") from exc

    def search(
        self,
        *,
        query: str,
        query_embedding: EmbeddingItem,
        top_k: int = DEFAULT_TOP_K,
        collection_name: str | None = None,
    ) -> list[RetrievedChunk]:
        """Run similarity search against indexed chunks."""

        if top_k <= 0:
            raise RetrievalServiceError("top_k must be greater than 0.")

        collection = self._get_collection(collection_name)
        try:
            results = collection.query(
                query_embeddings=[query_embedding.vector],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # pragma: no cover - Chroma raises library-specific errors
            raise RetrievalServiceError("Failed to query ChromaDB.") from exc

        return self._build_retrieved_chunks(
            results=results,
            collection_name=collection.name,
        )

    def _get_collection(self, collection_name: str | None = None) -> Collection:
        name = collection_name or self.collection_name
        return self.client.get_or_create_collection(name=name)

    @staticmethod
    def _pair_chunks_with_embeddings(
        chunks: list[KnowledgeChunk],
        embeddings: EmbeddingBatch,
    ) -> list[IndexedChunk]:
        paired: list[IndexedChunk] = []
        for chunk, embedding in zip(chunks, embeddings.items, strict=True):
            paired.append(IndexedChunk(chunk=chunk, vector=embedding.vector))
        return paired

    @staticmethod
    def _metadata_for_chunk(chunk: KnowledgeChunk) -> dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "source_path": chunk.source_path,
            "filename": chunk.filename,
            "category": chunk.category,
            "chunk_index": chunk.chunk_index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
        }

    @staticmethod
    def _build_retrieved_chunks(
        *,
        results: dict[str, Any],
        collection_name: str,
    ) -> list[RetrievedChunk]:
        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        distances = results.get("distances") or [[]]
        ids = results.get("ids") or [[]]

        retrieved: list[RetrievedChunk] = []
        for chunk_id, document, metadata, distance in zip(
            ids[0],
            documents[0],
            metadatas[0],
            distances[0] if distances else [None] * len(ids[0]),
            strict=False,
        ):
            metadata = metadata or {}
            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(metadata.get("chunk_id") or chunk_id),
                    text=str(document or ""),
                    source_path=str(metadata.get("source_path") or ""),
                    filename=str(metadata.get("filename") or ""),
                    category=str(metadata.get("category") or ""),
                    chunk_index=int(metadata.get("chunk_index") or 0),
                    start_char=int(metadata.get("start_char") or 0),
                    end_char=int(metadata.get("end_char") or 0),
                    similarity_distance=float(distance) if distance is not None else None,
                    collection_name=collection_name,
                )
            )

        return retrieved


def upsert_document_chunks(
    chunks: list[KnowledgeChunk],
    embeddings: EmbeddingBatch,
    *,
    store_path: Path = DEFAULT_VECTOR_STORE_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> None:
    """Convenience wrapper for indexing chunk embeddings."""

    store = ChromaRetrievalStore(
        store_path=store_path,
        collection_name=collection_name,
    )
    store.upsert_chunks(chunks, embeddings)


def search_chunks(
    *,
    query: str,
    query_embedding: EmbeddingItem,
    top_k: int = DEFAULT_TOP_K,
    store_path: Path = DEFAULT_VECTOR_STORE_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> list[RetrievedChunk]:
    """Convenience wrapper for similarity search."""

    store = ChromaRetrievalStore(
        store_path=store_path,
        collection_name=collection_name,
    )
    return store.search(
        query=query,
        query_embedding=query_embedding,
        top_k=top_k,
    )
