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
import json
import re
from functools import lru_cache

from ..llm_service import LLMResponse, OllamaService
from .embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingServiceError,
    OllamaEmbeddingService,
)
from .ingest import load_documents
from .retrieval import (
    DEFAULT_CANDIDATE_MULTIPLIER,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_MAX_SIMILARITY_DISTANCE,
    DEFAULT_TOP_K,
    RetrievalDiagnostics,
    RetrievalResult,
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
DOMAIN_KEYWORDS = {
    "generator",
    "honda",
    "eu2200it",
    "carburetor",
    "stator",
    "oil",
    "spark",
    "plug",
    "switch",
    "maintenance",
    "fuel",
    "engine",
}
LOW_SIGNAL_QUERY_TOKENS = {"yes", "no", "device", "problem", "issue", "help", "trouble"}


@dataclass(frozen=True)
class ProcedureRoute:
    document_id: str
    filename: str
    title: str
    trigger_terms: tuple[str, ...]


@dataclass(frozen=True)
class DomainGateDecision:
    allowed: bool
    reason: str


@dataclass(frozen=True)
class RetrievalDecision:
    use_retrieval: bool
    reason: str
    selected_document_id: str = ""
    selected_document_title: str = ""


@dataclass(frozen=True)
class RAGReply:
    """Response payload including retrieved context metadata."""

    llm_response: LLMResponse
    retrieved_chunks: list[RetrievedChunk]
    augmented_message: str
    used_retrieval: bool
    retrieval_diagnostics: RetrievalDiagnostics | None = None
    retrieval_reason: str = ""
    selected_document_id: str = ""
    selected_document_title: str = ""


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
        max_similarity_distance: float | None = DEFAULT_MAX_SIMILARITY_DISTANCE,
        candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
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
        self.max_similarity_distance = max_similarity_distance
        self.candidate_multiplier = candidate_multiplier
        self.vector_store_path = Path(vector_store_path)
        self.rag_instructions = rag_instructions

    async def generate_reply(
        self,
        *,
        message_text: str,
        session_id: str,
        conversation_history: str = "",
        llm_thread_history: str = "",
        attachment_context: str = "",
        vision_findings: dict[str, list[str]] | None = None,
        active_document_hint: str = "",
        active_document_id: str = "",
        retrieval_mode: str = "global",
        model: str | None = None,
        system_prompt: str | None = None,
        top_k: int | None = None,
    ) -> RAGReply:
        """Run the full retrieval-augmented generation workflow."""

        original_question = str(message_text).strip()
        if not original_question:
            raise RAGServiceError("User message cannot be empty.")
        final_request = self._build_final_request(original_question, vision_findings)

        if retrieval_mode == "disabled":
            llm_response = await self.llm_service.generate_reply(
                message_text=final_request,
                conversation_history=conversation_history,
                llm_thread_history=llm_thread_history,
                attachment_context=attachment_context,
                model=model,
                system_prompt=system_prompt,
            )
            return RAGReply(
                llm_response=llm_response,
                retrieved_chunks=[],
                augmented_message=final_request,
                used_retrieval=False,
                retrieval_diagnostics=None,
                retrieval_reason="retrieval_disabled",
            )

        retrieval_message = self._rewrite_follow_up_query(
            self._build_retrieval_query(original_question, vision_findings),
            active_document_hint=active_document_hint,
        )
        routed_document = self._route_document_by_intent(retrieval_message)
        selected_document_id = active_document_id or routed_document.document_id if routed_document else active_document_id
        selected_document_title = active_document_hint or (routed_document.title if routed_document else "")
        domain_gate = self._evaluate_domain_relevance(
            retrieval_message,
            routed_document=routed_document,
            active_document_id=active_document_id,
        )
        if not domain_gate.allowed:
            llm_response = await self.llm_service.generate_reply(
                message_text=final_request,
                conversation_history=conversation_history,
                llm_thread_history=llm_thread_history,
                attachment_context=attachment_context,
                model=model,
                system_prompt=system_prompt,
            )
            return RAGReply(
                llm_response=llm_response,
                retrieved_chunks=[],
                augmented_message=final_request,
                used_retrieval=False,
                retrieval_diagnostics=None,
                retrieval_reason=domain_gate.reason,
                selected_document_id=selected_document_id or "",
                selected_document_title=selected_document_title or "",
            )

        try:
            query_embedding = await self.embedding_service.embed_query_text(retrieval_message)
            retrieval_result = search_chunks(
                query=retrieval_message,
                query_embedding=query_embedding,
                top_k=top_k or self.top_k,
                neighbor_window=self.neighbor_window,
                max_similarity_distance=self.max_similarity_distance,
                candidate_multiplier=self.candidate_multiplier,
                document_id=selected_document_id or None,
                store_path=self.vector_store_path,
                collection_name=self.collection_name,
            )
            retrieved_chunks = retrieval_result.chunks
        except (EmbeddingServiceError, RetrievalServiceError) as exc:
            llm_response = await self.llm_service.generate_reply(
                message_text=final_request,
                conversation_history=conversation_history,
                llm_thread_history=llm_thread_history,
                attachment_context=attachment_context,
                model=model,
                system_prompt=system_prompt,
            )
            return RAGReply(
                llm_response=llm_response,
                retrieved_chunks=[],
                augmented_message=retrieval_message,
                used_retrieval=False,
                retrieval_diagnostics=None,
                retrieval_reason="retrieval_error",
                selected_document_id=selected_document_id or "",
                selected_document_title=selected_document_title or "",
            )

        retrieval_decision = self._evaluate_retrieval_confidence(
            retrieval_message,
            retrieval_result=retrieval_result,
            routed_document=routed_document,
        )
        if not retrieval_decision.use_retrieval:
            llm_response = await self.llm_service.generate_reply(
                message_text=final_request,
                conversation_history=conversation_history,
                llm_thread_history=llm_thread_history,
                attachment_context=attachment_context,
                model=model,
                system_prompt=system_prompt,
            )
            return RAGReply(
                llm_response=llm_response,
                retrieved_chunks=[],
                augmented_message=final_request,
                used_retrieval=False,
                retrieval_diagnostics=retrieval_result.diagnostics,
                retrieval_reason=retrieval_decision.reason,
                selected_document_id=retrieval_decision.selected_document_id,
                selected_document_title=retrieval_decision.selected_document_title,
            )

        if not retrieved_chunks:
            llm_response = await self.llm_service.generate_reply(
                message_text=final_request,
                conversation_history=conversation_history,
                llm_thread_history=llm_thread_history,
                attachment_context=attachment_context,
                model=model,
                system_prompt=system_prompt,
            )
            return RAGReply(
                llm_response=llm_response,
                retrieved_chunks=[],
                augmented_message=retrieval_message,
                used_retrieval=False,
                retrieval_diagnostics=retrieval_result.diagnostics,
                retrieval_reason="no_retrieval_results",
                selected_document_id=selected_document_id or "",
                selected_document_title=selected_document_title or "",
            )

        augmented_message = self._build_augmented_message(
            message_text=final_request,
            retrieved_chunks=retrieved_chunks,
        )
        selected_system_prompt = self._build_system_prompt(system_prompt)

        llm_response = await self.llm_service.generate_reply(
            message_text=augmented_message,
            conversation_history=conversation_history,
            llm_thread_history=llm_thread_history,
            attachment_context=attachment_context,
            model=model,
            system_prompt=selected_system_prompt,
        )

        return RAGReply(
            llm_response=llm_response,
            retrieved_chunks=retrieved_chunks,
            augmented_message=augmented_message,
            used_retrieval=True,
            retrieval_diagnostics=retrieval_result.diagnostics,
            retrieval_reason=retrieval_decision.reason,
            selected_document_id=retrieval_decision.selected_document_id,
            selected_document_title=retrieval_decision.selected_document_title,
        )

    @staticmethod
    def _build_retrieval_query(
        original_question: str,
        vision_findings: dict[str, list[str]] | None,
    ) -> str:
        if not vision_findings:
            return original_question
        return (
            f"Original user question:\n{original_question}\n\n"
            f"Structured visual findings:\n{json.dumps(vision_findings, ensure_ascii=True)}"
        )

    @staticmethod
    def _build_final_request(
        original_question: str,
        vision_findings: dict[str, list[str]] | None,
    ) -> str:
        if not vision_findings:
            return original_question
        return (
            f"Original user question:\n{original_question}\n\n"
            f"Structured visual findings from image analysis:\n"
            f"{json.dumps(vision_findings, ensure_ascii=True)}\n\n"
            "Use these findings as evidence, noting any listed uncertainty."
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

    def _evaluate_domain_relevance(
        self,
        message_text: str,
        *,
        routed_document: ProcedureRoute | None,
        active_document_id: str,
    ) -> DomainGateDecision:
        if routed_document is not None or active_document_id:
            return DomainGateDecision(True, "explicit_or_active_document")

        query_tokens = _tokenize(message_text)
        if not query_tokens:
            return DomainGateDecision(False, "empty_query")

        kb_tokens = self._knowledge_base_domain_tokens()
        overlap = query_tokens & kb_tokens
        if overlap:
            return DomainGateDecision(True, "domain_keyword_match")

        if len(query_tokens) <= 4 and query_tokens & LOW_SIGNAL_QUERY_TOKENS:
            return DomainGateDecision(False, "low_signal_out_of_domain")

        return DomainGateDecision(False, "no_domain_match")

    def _route_document_by_intent(self, message_text: str) -> ProcedureRoute | None:
        query_tokens = _tokenize(message_text)
        lowered = message_text.lower()
        best_route: ProcedureRoute | None = None
        best_score = 0.0

        for route in self._procedure_routes():
            trigger_tokens = {_normalize_token(term) for term in route.trigger_terms}
            overlap = query_tokens & trigger_tokens
            score = float(len(overlap))
            if route.title.lower() in lowered:
                score += 3.0
            if route.filename.lower().replace(".pdf", "") in lowered.replace("-", "_"):
                score += 2.0
            if score > best_score:
                best_score = score
                best_route = route

        if best_score >= 1.0:
            return best_route
        return None

    def _evaluate_retrieval_confidence(
        self,
        message_text: str,
        *,
        retrieval_result: RetrievalResult,
        routed_document: ProcedureRoute | None,
    ) -> RetrievalDecision:
        diagnostics = retrieval_result.diagnostics
        chunks = retrieval_result.chunks
        if not chunks:
            return RetrievalDecision(False, "no_chunks")

        top_chunk = chunks[0]
        query_tokens = _tokenize(message_text)
        lexical_overlap = len(query_tokens & _tokenize(top_chunk.text)) / max(1, len(query_tokens))
        distance = diagnostics.top_similarity_distance
        spread = diagnostics.document_spread

        if routed_document is not None:
            return RetrievalDecision(
                True,
                "explicit_task_route",
                selected_document_id=routed_document.document_id,
                selected_document_title=routed_document.title,
            )

        if distance is None:
            return RetrievalDecision(False, "missing_similarity_distance")
        if distance > 0.9:
            return RetrievalDecision(False, "low_similarity_confidence")
        if lexical_overlap < 0.2 and diagnostics.top_title_score < 0.2 and diagnostics.top_task_score < 0.2:
            return RetrievalDecision(False, "weak_lexical_and_task_match")
        if spread > 1 and diagnostics.top_similarity_distance is not None and diagnostics.second_similarity_distance is not None:
            if abs(diagnostics.second_similarity_distance - diagnostics.top_similarity_distance) < 0.03:
                return RetrievalDecision(False, "ambiguous_multi_document_match")

        return RetrievalDecision(
            True,
            "confident_retrieval_match",
            selected_document_id=top_chunk.document_id,
            selected_document_title=Path(top_chunk.filename).stem.replace("_", " ").replace("-", " ").strip(),
        )

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

    @staticmethod
    @lru_cache(maxsize=1)
    def _procedure_routes() -> tuple[ProcedureRoute, ...]:
        routes: list[ProcedureRoute] = []
        for document in load_documents(DEFAULT_VECTOR_STORE_PATH.parent / "knowledge_base"):
            title = document.text.splitlines()[0].strip() if document.text.splitlines() else document.filename
            trigger_terms = tuple(_tokenize(f"{title} {document.filename}"))
            routes.append(
                ProcedureRoute(
                    document_id=document.document_id,
                    filename=document.filename,
                    title=title,
                    trigger_terms=trigger_terms,
                )
            )
        return tuple(routes)

    @classmethod
    @lru_cache(maxsize=1)
    def _knowledge_base_domain_tokens(cls) -> set[str]:
        tokens = set(DOMAIN_KEYWORDS)
        for route in cls._procedure_routes():
            tokens.update(_tokenize(route.title))
            tokens.update(_tokenize(route.filename))
        return tokens


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]{2,}", str(text).lower())}


def _normalize_token(text: str) -> str:
    return str(text).strip().lower()
