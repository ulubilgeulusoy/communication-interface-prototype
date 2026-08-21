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
import re

import chromadb
from chromadb.api.models.Collection import Collection

from .embeddings import EmbeddingBatch, EmbeddingItem
from .ingest import KnowledgeChunk


DEFAULT_COLLECTION_NAME = "knowledge_chunks"
DEFAULT_TOP_K = 5
DEFAULT_VECTOR_STORE_PATH = Path(__file__).resolve().parent.parent.parent / "vector_store"
DEFAULT_MAX_SIMILARITY_DISTANCE = 1.2
DEFAULT_CANDIDATE_MULTIPLIER = 4
TOKEN_PATTERN = re.compile(r"[a-z0-9]{2,}")


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
    document_id: str
    text: str
    source_path: str
    filename: str
    category: str
    chunk_index: int
    start_char: int
    end_char: int
    similarity_distance: float | None
    collection_name: str
    lexical_score: float = 0.0
    title_score: float = 0.0
    task_score: float = 0.0


@dataclass(frozen=True)
class RetrievalDiagnostics:
    query: str
    requested_top_k: int
    raw_candidate_count: int
    filtered_candidate_count: int
    returned_chunk_count: int
    max_similarity_distance: float | None
    top_document_id: str = ""
    top_filename: str = ""
    top_similarity_distance: float | None = None
    second_similarity_distance: float | None = None
    top_lexical_score: float = 0.0
    top_title_score: float = 0.0
    top_task_score: float = 0.0
    document_spread: int = 0


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    diagnostics: RetrievalDiagnostics


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
        indexed_by_document = self._group_indexed_chunks_by_document(indexed_chunks)

        for document_id, document_chunks in indexed_by_document.items():
            try:
                collection.delete(where={"document_id": document_id})
                collection.upsert(
                    ids=[item.chunk.chunk_id for item in document_chunks],
                    embeddings=[item.vector for item in document_chunks],
                    documents=[item.chunk.text for item in document_chunks],
                    metadatas=[self._metadata_for_chunk(item.chunk) for item in document_chunks],
                )
            except Exception as exc:  # pragma: no cover - Chroma raises library-specific errors
                raise RetrievalServiceError(
                    f"Failed to refresh chunks in ChromaDB for document '{document_id}'."
                ) from exc

    def search(
        self,
        *,
        query: str,
        query_embedding: EmbeddingItem,
        top_k: int = DEFAULT_TOP_K,
        neighbor_window: int = 1,
        max_similarity_distance: float | None = DEFAULT_MAX_SIMILARITY_DISTANCE,
        candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
        document_id: str | None = None,
        collection_name: str | None = None,
    ) -> RetrievalResult:
        """Run similarity search against indexed chunks."""

        if top_k <= 0:
            raise RetrievalServiceError("top_k must be greater than 0.")
        if neighbor_window < 0:
            raise RetrievalServiceError("neighbor_window must be 0 or greater.")
        if max_similarity_distance is not None and max_similarity_distance < 0:
            raise RetrievalServiceError("max_similarity_distance must be 0 or greater.")
        if candidate_multiplier <= 0:
            raise RetrievalServiceError("candidate_multiplier must be greater than 0.")

        collection = self._get_collection(collection_name)
        try:
            results = collection.query(
                query_embeddings=[query_embedding.vector],
                n_results=max(top_k, top_k * candidate_multiplier),
                include=["documents", "metadatas", "distances"],
                where=self._build_where_filter(document_id=document_id),
            )
        except Exception as exc:  # pragma: no cover - Chroma raises library-specific errors
            raise RetrievalServiceError("Failed to query ChromaDB.") from exc

        raw_candidates = self._build_retrieved_chunks(
            results=results,
            collection_name=collection.name,
        )
        filtered_candidates = self._filter_by_similarity_distance(
            raw_candidates,
            max_similarity_distance=max_similarity_distance,
        )
        ranked_candidates = self._rerank_candidates(
            filtered_candidates,
            query=query,
        )[:top_k]
        diagnostics = RetrievalDiagnostics(
            query=query,
            requested_top_k=top_k,
            raw_candidate_count=len(raw_candidates),
            filtered_candidate_count=len(filtered_candidates),
            returned_chunk_count=len(ranked_candidates),
            max_similarity_distance=max_similarity_distance,
            top_document_id=ranked_candidates[0].document_id if ranked_candidates else "",
            top_filename=ranked_candidates[0].filename if ranked_candidates else "",
            top_similarity_distance=ranked_candidates[0].similarity_distance if ranked_candidates else None,
            second_similarity_distance=(
                ranked_candidates[1].similarity_distance if len(ranked_candidates) > 1 else None
            ),
            top_lexical_score=ranked_candidates[0].lexical_score if ranked_candidates else 0.0,
            top_title_score=ranked_candidates[0].title_score if ranked_candidates else 0.0,
            top_task_score=ranked_candidates[0].task_score if ranked_candidates else 0.0,
            document_spread=len({chunk.document_id for chunk in ranked_candidates}),
        )
        if not ranked_candidates or neighbor_window == 0:
            return RetrievalResult(chunks=ranked_candidates, diagnostics=diagnostics)

        expanded_chunks = self._expand_with_neighbors(
            collection=collection,
            retrieved=ranked_candidates,
            neighbor_window=neighbor_window,
        )
        return RetrievalResult(chunks=expanded_chunks, diagnostics=diagnostics)

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
    def _group_indexed_chunks_by_document(
        indexed_chunks: list[IndexedChunk],
    ) -> dict[str, list[IndexedChunk]]:
        grouped: dict[str, list[IndexedChunk]] = {}
        for item in indexed_chunks:
            grouped.setdefault(item.chunk.document_id, []).append(item)
        return grouped

    @staticmethod
    def _metadata_for_chunk(chunk: KnowledgeChunk) -> dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "source_path": chunk.source_path,
            "filename": chunk.filename,
            "category": chunk.category,
            "chunk_index": chunk.chunk_index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
        }

    def _expand_with_neighbors(
        self,
        *,
        collection: Collection,
        retrieved: list[RetrievedChunk],
        neighbor_window: int,
    ) -> list[RetrievedChunk]:
        ordered: list[RetrievedChunk] = []
        seen_chunk_ids: set[str] = set()

        for chunk in retrieved:
            for neighbor in self._fetch_neighbors(
                collection=collection,
                document_id=chunk.document_id,
                center_index=chunk.chunk_index,
                neighbor_window=neighbor_window,
            ):
                if neighbor.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(neighbor.chunk_id)
                ordered.append(neighbor)

        ordered.sort(key=lambda item: (item.source_path, item.chunk_index, item.chunk_id))
        return ordered

    @staticmethod
    def _filter_by_similarity_distance(
        retrieved: list[RetrievedChunk],
        *,
        max_similarity_distance: float | None,
    ) -> list[RetrievedChunk]:
        if max_similarity_distance is None:
            return retrieved

        filtered: list[RetrievedChunk] = []
        for chunk in retrieved:
            distance = chunk.similarity_distance
            if distance is None:
                continue
            if distance <= max_similarity_distance:
                filtered.append(chunk)
        return filtered

    @staticmethod
    def _rerank_candidates(
        retrieved: list[RetrievedChunk],
        *,
        query: str,
    ) -> list[RetrievedChunk]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return sorted(
                retrieved,
                key=lambda chunk: (
                    chunk.similarity_distance if chunk.similarity_distance is not None else float("inf")
                ),
            )

        scored: list[RetrievedChunk] = []
        for chunk in retrieved:
            lexical_score = _lexical_overlap_score(
                query_tokens,
                " ".join([chunk.filename, chunk.category, chunk.text]),
            )
            title_score = _title_overlap_score(query_tokens, chunk.filename, chunk.text)
            task_score = _task_overlap_score(query_tokens, chunk.text)
            scored.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    source_path=chunk.source_path,
                    filename=chunk.filename,
                    category=chunk.category,
                    chunk_index=chunk.chunk_index,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    similarity_distance=chunk.similarity_distance,
                    collection_name=chunk.collection_name,
                    lexical_score=lexical_score,
                    title_score=title_score,
                    task_score=task_score,
                )
            )

        def ranking_key(chunk: RetrievedChunk) -> tuple[float, float, float, float, int]:
            combined_score = chunk.lexical_score + (chunk.title_score * 2.5) + (chunk.task_score * 2.0)
            distance = chunk.similarity_distance if chunk.similarity_distance is not None else float("inf")
            step_match = 1 if re.search(r"(?i)\bstep\s+\d+\b", chunk.text) else 0
            return (-combined_score, -chunk.title_score, distance, -chunk.task_score, -step_match)

        return sorted(scored, key=ranking_key)

    def _fetch_neighbors(
        self,
        *,
        collection: Collection,
        document_id: str,
        center_index: int,
        neighbor_window: int,
    ) -> list[RetrievedChunk]:
        min_index = max(0, center_index - neighbor_window)
        max_index = center_index + neighbor_window
        try:
            results = collection.get(
                where={
                    "$and": [
                        {"document_id": document_id},
                        {"chunk_index": {"$gte": min_index}},
                        {"chunk_index": {"$lte": max_index}},
                    ]
                },
                include=["documents", "metadatas"],
            )
        except Exception as exc:  # pragma: no cover
            raise RetrievalServiceError("Failed to fetch neighboring chunks.") from exc

        return self._build_retrieved_chunks(
            results=results,
            collection_name=collection.name,
        )

    @staticmethod
    def _build_retrieved_chunks(
        *,
        results: dict[str, Any],
        collection_name: str,
    ) -> list[RetrievedChunk]:
        documents = _normalize_result_rows(results.get("documents"))
        metadatas = _normalize_result_rows(results.get("metadatas"))
        distances = _normalize_result_rows(results.get("distances"))
        ids = _normalize_result_rows(results.get("ids"))

        if not ids:
            return []

        first_ids = ids[0]
        first_documents = documents[0] if documents else [None] * len(first_ids)
        first_metadatas = metadatas[0] if metadatas else [None] * len(first_ids)
        first_distances = distances[0] if distances else [None] * len(first_ids)

        retrieved: list[RetrievedChunk] = []
        for chunk_id, document, metadata, distance in zip(
            first_ids,
            first_documents,
            first_metadatas,
            first_distances,
            strict=False,
        ):
            metadata = metadata or {}
            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(metadata.get("chunk_id") or chunk_id),
                    document_id=str(metadata.get("document_id") or ""),
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

    @staticmethod
    def _build_where_filter(*, document_id: str | None) -> dict[str, Any] | None:
        cleaned_document_id = str(document_id or "").strip()
        if not cleaned_document_id:
            return None
        return {"document_id": cleaned_document_id}


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
    neighbor_window: int = 1,
    max_similarity_distance: float | None = DEFAULT_MAX_SIMILARITY_DISTANCE,
    candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
    document_id: str | None = None,
    store_path: Path = DEFAULT_VECTOR_STORE_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> RetrievalResult:
    """Convenience wrapper for similarity search."""

    store = ChromaRetrievalStore(
        store_path=store_path,
        collection_name=collection_name,
    )
    return store.search(
        query=query,
        query_embedding=query_embedding,
        top_k=top_k,
        neighbor_window=neighbor_window,
        max_similarity_distance=max_similarity_distance,
        candidate_multiplier=candidate_multiplier,
        document_id=document_id,
    )


def _normalize_result_rows(value: Any) -> list[list[Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        return []
    if not value:
        return []
    if isinstance(value[0], list):
        return value
    return [value]


def _tokenize(text: str) -> set[str]:
    return {match.group(0) for match in TOKEN_PATTERN.finditer(str(text).lower())}


def _lexical_overlap_score(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    chunk_tokens = _tokenize(text)
    if not chunk_tokens:
        return 0.0
    return len(query_tokens & chunk_tokens) / len(query_tokens)


def _title_overlap_score(query_tokens: set[str], filename: str, text: str) -> float:
    if not query_tokens:
        return 0.0
    title_tokens = _tokenize(Path(filename).stem.replace("_", " ").replace("-", " "))
    header_tokens = _tokenize("\n".join(str(text).splitlines()[:3]))
    combined = title_tokens | header_tokens
    if not combined:
        return 0.0
    return len(query_tokens & combined) / len(query_tokens)


def _task_overlap_score(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    lowered = str(text).lower()
    matched = {
        token
        for token in query_tokens
        if token in lowered and token not in {"the", "and", "for", "with", "this", "that", "what", "does"}
    }
    return len(matched) / len(query_tokens)
