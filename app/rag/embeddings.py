"""Embedding utilities backed by the local Ollama server.

The final version of this module will call Ollama's embeddings endpoint and
return vector representations for knowledge chunks and user queries.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingRequest:
    """Input payload for a future Ollama embedding request."""

    text: str
    model: str = "nomic-embed-text"


def embed_texts(requests: list[EmbeddingRequest]) -> list[list[float]]:
    """Return placeholder embeddings for the provided texts."""

    return [[] for _ in requests]

