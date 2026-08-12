"""RAG orchestration layer for the communication prototype.

This service will sit between retrieval and the existing Ollama chat service,
combining retrieved context with the current user message before generation.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..llm_service import LLMResponse, OllamaService
from .retrieval import RetrievedChunk, search_chunks


@dataclass(frozen=True)
class RAGReply:
    """Response payload including retrieved context metadata."""

    llm_response: LLMResponse
    retrieved_chunks: list[RetrievedChunk]


class RAGService:
    """Thin placeholder wrapper around retrieval and the existing LLM service."""

    def __init__(self, llm_service: OllamaService) -> None:
        self.llm_service = llm_service

    async def generate_reply(self, *, message_text: str, session_id: str) -> RAGReply:
        """Retrieve supporting context and defer generation to the LLM service."""

        retrieved_chunks = search_chunks(message_text)
        context_text = "\n\n".join(chunk.text for chunk in retrieved_chunks if chunk.text)
        prompt = message_text if not context_text else f"Context:\n{context_text}\n\nUser request:\n{message_text}"
        llm_response = await self.llm_service.generate_reply(
            message_text=prompt,
            conversation_history="",
        )
        return RAGReply(llm_response=llm_response, retrieved_chunks=retrieved_chunks)

