"""RAG orchestration layer for the communication prototype.

This module coordinates:

- embedding the current user message
- retrieving relevant domain chunks from ChromaDB
- building a context block from retrieved knowledge
- delegating the final response generation to the existing Ollama chat service

The actual LLM inference remains inside ``app.llm_service``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from ..llm_service import LLMResponse, OllamaService
from .embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingServiceError,
    OllamaEmbeddingService,
)
from .retrieval import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_TOP_K,
    DEFAULT_VECTOR_STORE_PATH,
    RetrievedChunk,
    RetrievalServiceError,
    search_chunks,
)


DEFAULT_RAG_INSTRUCTIONS = (
    "Use the retrieved domain knowledge when it is relevant. "
    "If the retrieved context is incomplete or not applicable, say so clearly "
    "and avoid inventing unsupported facts. "
    "For procedures, prefer the retrieved steps exactly as written and call out "
    "when a step sequence appears incomplete."
)


class RAGServiceError(Exception):
    """Raised when the RAG workflow cannot complete."""


STEP_REFERENCE_PATTERN = re.compile(r"\bstep\s+\d+\b", re.IGNORECASE)
DOCUMENT_REFERENCE_PATTERN = re.compile(r'"[^"]+"|\'[^\']+\'', re.IGNORECASE)
FOLLOW_UP_PREFIXES = (
    "what is",
    "what does",
    "show me",
    "tell me",
    "and",
)


@dataclass(frozen=True)
class RAGReply:
    """Response payload including retrieved context metadata."""

    llm_response: LLMResponse
    retrieved_chunks: list[RetrievedChunk]
    augmented_message: str
    used_retrieval: bool


class RAGService:
    """Coordinate embedding, retrieval, and LLM response generation."""

    def __init__(
        self,
        llm_service: OllamaService,
        *,
        embedding_service: OllamaEmbeddingService | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        top_k: int = 7,
        neighbor_window: int = 1,
        vector_store_path: Path = DEFAULT_VECTOR_STORE_PATH,
        rag_instructions: str = DEFAULT_RAG_INSTRUCTIONS,
    ) -> None:
        self.llm_service = llm_service
        self.embedding_service = embedding_service or OllamaEmbeddingService(
            model=DEFAULT_EMBEDDING_MODEL
        )
        self.collection_name = collection_name
        self.top_k = top_k
        self.neighbor_window = neighbor_window
        self.vector_store_path = Path(vector_store_path)
        self.rag_instructions = rag_instructions

    async def generate_reply(
        self,
        *,
        message_text: str,
        session_id: str,
        conversation_history: str = "",
        llm_thread_history: str = "",
        active_document_hint: str = "",
        active_document_id: str = "",
        model: str | None = None,
        system_prompt: str | None = None,
        top_k: int | None = None,
    ) -> RAGReply:
        """Run the full retrieval-augmented generation workflow."""

        cleaned_message = str(message_text).strip()
        if not cleaned_message:
            raise RAGServiceError("User message cannot be empty.")

        retrieval_message = self._rewrite_follow_up_query(
            cleaned_message,
            active_document_hint=active_document_hint,
        )

        try:
            query_embedding = await self.embedding_service.embed_text(retrieval_message)
            retrieved_chunks = search_chunks(
                query=retrieval_message,
                query_embedding=query_embedding,
                top_k=top_k or self.top_k,
                neighbor_window=self.neighbor_window,
                document_id=active_document_id or None,
                store_path=self.vector_store_path,
                collection_name=self.collection_name,
            )
        except (EmbeddingServiceError, RetrievalServiceError) as exc:
            llm_response = await self.llm_service.generate_reply(
                message_text=cleaned_message,
                conversation_history=conversation_history,
                llm_thread_history=llm_thread_history,
                model=model,
                system_prompt=system_prompt,
            )
            return RAGReply(
                llm_response=llm_response,
                retrieved_chunks=[],
                augmented_message=retrieval_message,
                used_retrieval=False,
            )

        if not retrieved_chunks:
            llm_response = await self.llm_service.generate_reply(
                message_text=cleaned_message,
                conversation_history=conversation_history,
                llm_thread_history=llm_thread_history,
                model=model,
                system_prompt=system_prompt,
            )
            return RAGReply(
                llm_response=llm_response,
                retrieved_chunks=[],
                augmented_message=retrieval_message,
                used_retrieval=False,
            )

        augmented_message = self._build_augmented_message(
            message_text=retrieval_message,
            retrieved_chunks=retrieved_chunks,
        )
        selected_system_prompt = self._build_system_prompt(system_prompt)

        llm_response = await self.llm_service.generate_reply(
            message_text=augmented_message,
            conversation_history=conversation_history,
            llm_thread_history=llm_thread_history,
            model=model,
            system_prompt=selected_system_prompt,
        )

        return RAGReply(
            llm_response=llm_response,
            retrieved_chunks=retrieved_chunks,
            augmented_message=augmented_message,
            used_retrieval=True,
        )

    def _build_augmented_message(
        self,
        *,
        message_text: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> str:
        if not retrieved_chunks:
            return f"Domain knowledge:\nNone retrieved.\n\nUser request:\n{message_text}"

        context_lines: list[str] = []
        for index, chunk in enumerate(retrieved_chunks, start=1):
            header = (
                f"[{index}] category={chunk.category} "
                f"source={chunk.filename} "
                f"chunk={chunk.chunk_index}"
            )
            context_lines.append(f"{header}\n{chunk.text}")

        context_block = "\n\n".join(context_lines)
        return f"Domain knowledge:\n{context_block}\n\nUser request:\n{message_text}"

    def _build_system_prompt(self, system_prompt: str | None) -> str:
        base_prompt = system_prompt or self.llm_service.system_prompt
        return f"{base_prompt} {self.rag_instructions}".strip()

    def _rewrite_follow_up_query(
        self,
        message_text: str,
        *,
        active_document_hint: str,
    ) -> str:
        cleaned_hint = str(active_document_hint).strip()
        if not cleaned_hint:
            return message_text

        lowered = message_text.lower()
        has_step_reference = bool(STEP_REFERENCE_PATTERN.search(message_text))
        has_document_reference = bool(DOCUMENT_REFERENCE_PATTERN.search(message_text))
        looks_like_follow_up = lowered.startswith(FOLLOW_UP_PREFIXES)

        if has_step_reference and not has_document_reference:
            return f'In "{cleaned_hint}", {message_text}'

        if looks_like_follow_up and not has_document_reference and len(message_text.split()) <= 10:
            return f'About "{cleaned_hint}", {message_text}'

        return message_text
