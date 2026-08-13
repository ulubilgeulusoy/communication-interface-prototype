"""Embedding utilities backed by the local Ollama API.

This module is intentionally independent from retrieval and generation. It only
knows how to turn text into vectors by calling Ollama's embeddings endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx


DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DOCUMENT_EMBEDDING_PREFIX = "search_document: "
QUERY_EMBEDDING_PREFIX = "search_query: "


class EmbeddingServiceError(Exception):
    """Raised when Ollama embeddings cannot be generated."""


@dataclass(frozen=True)
class EmbeddingItem:
    """Embedding output for a single input string."""

    text: str
    vector: list[float]
    model: str
    index: int


@dataclass(frozen=True)
class EmbeddingBatch:
    """Embedding output for one Ollama embedding request."""

    model: str
    items: list[EmbeddingItem]

    @property
    def vectors(self) -> list[list[float]]:
        """Convenience accessor for the raw vectors."""

        return [item.vector for item in self.items]


class OllamaEmbeddingService:
    """Generate embeddings through the local Ollama API."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        model: str = DEFAULT_EMBEDDING_MODEL,
        timeout: float = 20.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def embed_text(self, text: str, *, model: str | None = None) -> EmbeddingItem:
        """Embed a single text string."""

        batch = await self.embed_texts([text], model=model)
        return batch.items[0]

    async def embed_document_text(
        self,
        text: str,
        *,
        model: str | None = None,
    ) -> EmbeddingItem:
        """Embed one retrieval document using Nomic's document prefix."""

        batch = await self.embed_document_texts([text], model=model)
        return batch.items[0]

    async def embed_query_text(
        self,
        text: str,
        *,
        model: str | None = None,
    ) -> EmbeddingItem:
        """Embed one retrieval query using Nomic's query prefix."""

        batch = await self.embed_query_texts([text], model=model)
        return batch.items[0]

    async def embed_texts(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        truncate: bool = True,
    ) -> EmbeddingBatch:
        """Embed a batch of text strings."""

        cleaned_texts = [self._clean_text(text) for text in texts]
        if not cleaned_texts:
            raise EmbeddingServiceError("At least one text input is required.")
        if any(not text for text in cleaned_texts):
            raise EmbeddingServiceError("Embedding inputs cannot be empty.")

        selected_model = model or self.model
        payload = {
            "model": selected_model,
            "input": cleaned_texts,
            "truncate": truncate,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/api/embed", json=payload)
                response.raise_for_status()
        except httpx.ConnectError as exc:
            raise EmbeddingServiceError("Unable to reach the local Ollama server.") from exc
        except httpx.TimeoutException as exc:
            raise EmbeddingServiceError("Ollama embedding request timed out.") from exc
        except httpx.HTTPStatusError as exc:
            detail = self._extract_error_detail(exc.response)
            raise EmbeddingServiceError(detail) from exc
        except httpx.RequestError as exc:
            raise EmbeddingServiceError("Ollama embedding request failed.") from exc

        data = response.json()
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(cleaned_texts):
            raise EmbeddingServiceError("Ollama returned an invalid embeddings payload.")

        items: list[EmbeddingItem] = []
        for index, (text, vector) in enumerate(zip(cleaned_texts, embeddings, strict=False)):
            if not isinstance(vector, list) or not all(isinstance(value, (int, float)) for value in vector):
                raise EmbeddingServiceError("Ollama returned a malformed embedding vector.")
            items.append(
                EmbeddingItem(
                    text=text,
                    vector=[float(value) for value in vector],
                    model=selected_model,
                    index=index,
                )
            )

        return EmbeddingBatch(model=selected_model, items=items)

    async def embed_document_texts(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        truncate: bool = True,
    ) -> EmbeddingBatch:
        """Embed retrieval documents using Nomic's document-task prefix."""

        return await self._embed_prefixed_texts(
            texts=texts,
            prefixer=self._prefix_document_text,
            model=model,
            truncate=truncate,
        )

    async def embed_query_texts(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        truncate: bool = True,
    ) -> EmbeddingBatch:
        """Embed retrieval queries using Nomic's query-task prefix."""

        return await self._embed_prefixed_texts(
            texts=texts,
            prefixer=self._prefix_query_text,
            model=model,
            truncate=truncate,
        )

    async def _embed_prefixed_texts(
        self,
        *,
        texts: list[str],
        prefixer,
        model: str | None,
        truncate: bool,
    ) -> EmbeddingBatch:
        prefixed_texts = [prefixer(text) for text in texts]
        try:
            return await self.embed_texts(
                prefixed_texts,
                model=model,
                truncate=truncate,
            )
        except EmbeddingServiceError as exc:
            # Some local Ollama/model combinations may reject retrieval-task prefixes.
            # Fall back to raw text so the app continues to function.
            if not self._should_fallback_to_raw_text(exc):
                raise
            return await self.embed_texts(
                texts,
                model=model,
                truncate=truncate,
            )

    @staticmethod
    def _clean_text(text: str) -> str:
        return str(text).strip()

    @classmethod
    def _prefix_document_text(cls, text: str) -> str:
        return f"{DOCUMENT_EMBEDDING_PREFIX}{cls._clean_text(text)}"

    @classmethod
    def _prefix_query_text(cls, text: str) -> str:
        return f"{QUERY_EMBEDDING_PREFIX}{cls._clean_text(text)}"

    @staticmethod
    def _should_fallback_to_raw_text(exc: EmbeddingServiceError) -> bool:
        message = str(exc).lower()
        fallback_markers = (
            "invalid",
            "malformed",
            "timed out",
            "failed with status 4",
            "failed with status 5",
            "not found",
        )
        return any(marker in message for marker in fallback_markers)

    @staticmethod
    def _extract_error_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return f"Ollama embedding request failed with status {response.status_code}."

        error_text = payload.get("error")
        if isinstance(error_text, str) and error_text.strip():
            return error_text.strip()

        return f"Ollama embedding request failed with status {response.status_code}."
