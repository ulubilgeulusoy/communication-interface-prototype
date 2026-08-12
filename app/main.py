from __future__ import annotations

import json
from pathlib import Path
import re

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .db import (
    init_db,
    list_llm_interactions,
    list_messages,
    list_recent_llm_interactions,
    list_recent_messages,
    log_llm_interaction,
    log_message,
    mark_delivered,
    sanitize_session_id,
    utc_now_iso,
)
from .experiment import assign_condition
from .llm_service import DEFAULT_MODEL, OllamaService, OllamaServiceError
from .rag.embeddings import EmbeddingServiceError
from .rag.index_knowledge import (
    KNOWLEDGE_BASE_PATH,
    run_indexing,
)
from .rag.ingest import load_documents
from .rag.rag_service import RAGService
from .rag.retrieval import DEFAULT_COLLECTION_NAME, DEFAULT_VECTOR_STORE_PATH, RetrievalServiceError


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "frontend"

app = FastAPI(title="Two-User Communication Prototype")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
ollama_service = OllamaService()
rag_service = RAGService(ollama_service)
LLM_HISTORY_LIMIT = 12
LLM_THREAD_HISTORY_LIMIT = 6


class LLMRequest(BaseModel):
    session_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    message_text: str = Field(min_length=1)
    model: str | None = None


class LLMReply(BaseModel):
    session_id: str
    user_id: str
    model: str
    output_text: str
    timestamp: str
    retrieved_chunks: list[dict] = Field(default_factory=list)
    used_retrieval: bool = False


class SessionHistory(BaseModel):
    session_id: str
    messages: list[dict]
    llm_interactions: list[dict]


class RAGReindexReply(BaseModel):
    indexed_files: int
    indexed_chunks: int
    collection_name: str
    embedding_model: str
    knowledge_base_path: str


QUOTED_TITLE_PATTERN = re.compile(r'"([^"]{3,})"|\'([^\']{3,})\'')


def build_conversation_history(session_id: str, limit: int = LLM_HISTORY_LIMIT) -> str:
    recent_messages = list_recent_messages(session_id, limit=limit)
    if not recent_messages:
        return ""

    lines: list[str] = []
    for message in recent_messages:
        sender = message["sender"]
        receiver = message["receiver"]
        content = str(message["content"]).strip()
        if not content:
            continue
        lines.append(f"{sender} to {receiver}: {content}")

    return "\n".join(lines)


def build_llm_thread_history(
    session_id: str,
    *,
    user_id: str,
    limit: int = LLM_THREAD_HISTORY_LIMIT,
) -> str:
    recent_interactions = list_recent_llm_interactions(
        session_id,
        user_id=user_id,
        limit=limit,
    )
    if not recent_interactions:
        return ""

    lines: list[str] = []
    for interaction in recent_interactions:
        prompt_text = str(interaction["input_text"]).strip()
        reply_text = str(interaction["output_text"]).strip()
        if prompt_text:
            lines.append(f"User to LLM: {prompt_text}")
        if reply_text:
            lines.append(f"LLM to user: {reply_text}")

    return "\n".join(lines)


def infer_active_document_context(
    session_id: str,
    *,
    user_id: str,
    limit: int = LLM_THREAD_HISTORY_LIMIT,
) -> tuple[str, str]:
    recent_interactions = list_recent_llm_interactions(
        session_id,
        user_id=user_id,
        limit=limit,
    )
    for interaction in reversed(recent_interactions):
        prompt_text = str(interaction.get("input_text") or "").strip()
        quoted_title = _extract_quoted_title(prompt_text)
        retrieved_sources = _parse_retrieved_sources(interaction.get("retrieved_sources_json"))
        if retrieved_sources:
            matched_source = _find_matching_source(retrieved_sources, quoted_title)
            if matched_source:
                return matched_source

            first_source = retrieved_sources[0]
            filename = str(first_source.get("filename") or "").strip()
            document_id = str(first_source.get("document_id") or "").strip()
            if filename or document_id:
                return (
                    Path(filename).stem.replace("_", " ").replace("-", " ").strip(),
                    document_id,
                )

        if quoted_title:
            return quoted_title, ""

    return "", ""


def _parse_retrieved_sources(raw_value: object) -> list[dict]:
    if not raw_value:
        return []
    try:
        parsed = json.loads(str(raw_value))
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _extract_quoted_title(text: str) -> str:
    match = QUOTED_TITLE_PATTERN.search(text)
    if not match:
        return ""
    return str(match.group(1) or match.group(2) or "").strip()


def _find_matching_source(retrieved_sources: list[dict], quoted_title: str) -> tuple[str, str] | None:
    cleaned_title = quoted_title.strip().lower()
    if cleaned_title:
        for source in retrieved_sources:
            filename = str(source.get("filename") or "").strip()
            document_id = str(source.get("document_id") or "").strip()
            stem = Path(filename).stem.replace("_", " ").replace("-", " ").strip().lower()
            if stem == cleaned_title:
                return (
                    Path(filename).stem.replace("_", " ").replace("-", " ").strip(),
                    document_id,
                )

    for source in retrieved_sources:
        filename = str(source.get("filename") or "").strip()
        document_id = str(source.get("document_id") or "").strip()
        if filename or document_id:
            return (
                Path(filename).stem.replace("_", " ").replace("-", " ").strip(),
                document_id,
            )

    return None


class ConnectionManager:
    def __init__(self) -> None:
        self.active: dict[str, WebSocket] = {}

    async def connect(self, user_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active[user_id] = websocket

    def disconnect(self, user_id: str) -> None:
        self.active.pop(user_id, None)

    async def send_to_user(self, user_id: str, payload: dict) -> bool:
        websocket = self.active.get(user_id)
        if websocket is None:
            return False

        await websocket.send_json(payload)
        return True


manager = ConnectionManager()


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/condition/{session_id}")
def get_condition(session_id: str) -> dict[str, str]:
    safe_session_id = sanitize_session_id(session_id)
    return {
        "session_id": safe_session_id,
        "experimental_condition": assign_condition(safe_session_id),
    }


@app.get("/api/messages")
def get_messages(session_id: str = Query(..., min_length=1)) -> list[dict]:
    safe_session_id = sanitize_session_id(session_id)
    return list_messages(safe_session_id)


@app.get("/api/session/{session_id}/history", response_model=SessionHistory)
def get_session_history(session_id: str) -> SessionHistory:
    safe_session_id = sanitize_session_id(session_id)
    return SessionHistory(
        session_id=safe_session_id,
        messages=list_messages(safe_session_id),
        llm_interactions=list_llm_interactions(safe_session_id),
    )


@app.post("/api/llm/message", response_model=LLMReply)
async def ask_llm(payload: LLMRequest) -> LLMReply:
    safe_session_id = sanitize_session_id(payload.session_id)
    timestamp = utc_now_iso()
    conversation_history = build_conversation_history(safe_session_id)
    llm_thread_history = build_llm_thread_history(
        safe_session_id,
        user_id=payload.user_id,
    )
    active_document_hint, active_document_id = infer_active_document_context(
        safe_session_id,
        user_id=payload.user_id,
    )

    try:
        rag_reply = await rag_service.generate_reply(
            message_text=payload.message_text,
            session_id=safe_session_id,
            conversation_history=conversation_history,
            llm_thread_history=llm_thread_history,
            active_document_hint=active_document_hint,
            active_document_id=active_document_id,
            model=payload.model,
        )
    except OllamaServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    retrieved_chunks_payload = [
        {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "source_path": chunk.source_path,
            "filename": chunk.filename,
            "category": chunk.category,
            "chunk_index": chunk.chunk_index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "similarity_distance": chunk.similarity_distance,
            "collection_name": chunk.collection_name,
        }
        for chunk in rag_reply.retrieved_chunks
    ]

    log_llm_interaction(
        timestamp=timestamp,
        session_id=safe_session_id,
        user_id=payload.user_id,
        model=rag_reply.llm_response.model or payload.model or DEFAULT_MODEL,
        input_text=payload.message_text,
        output_text=rag_reply.llm_response.output_text,
        retrieved_sources_json=json.dumps(retrieved_chunks_payload),
    )

    return LLMReply(
        session_id=safe_session_id,
        user_id=payload.user_id,
        model=rag_reply.llm_response.model,
        output_text=rag_reply.llm_response.output_text,
        timestamp=timestamp,
        retrieved_chunks=retrieved_chunks_payload,
        used_retrieval=rag_reply.used_retrieval,
    )


@app.post("/api/rag/reindex", response_model=RAGReindexReply)
async def reindex_knowledge_base() -> RAGReindexReply:
    try:
        indexed_chunks = await run_indexing(
            knowledge_base_path=KNOWLEDGE_BASE_PATH,
            vector_store_path=DEFAULT_VECTOR_STORE_PATH,
            collection_name=DEFAULT_COLLECTION_NAME,
            embedding_model=rag_service.embedding_service.model,
            chunk_size=1000,
            chunk_overlap=200,
        )
    except (EmbeddingServiceError, RetrievalServiceError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    indexed_files = len(load_documents(KNOWLEDGE_BASE_PATH))
    return RAGReindexReply(
        indexed_files=indexed_files,
        indexed_chunks=indexed_chunks,
        collection_name=DEFAULT_COLLECTION_NAME,
        embedding_model=rag_service.embedding_service.model,
        knowledge_base_path=str(KNOWLEDGE_BASE_PATH),
    )


@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str) -> None:
    await manager.connect(user_id, websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)

            receiver = str(data["receiver"])
            content = str(data["content"])
            session_id = sanitize_session_id(str(data["session_id"]))
            experimental_condition = str(
                data.get("experimental_condition") or assign_condition(session_id)
            )
            sent_timestamp = utc_now_iso()

            message_id = log_message(
                sender=user_id,
                receiver=receiver,
                content=content,
                sent_timestamp=sent_timestamp,
                session_id=session_id,
                experimental_condition=experimental_condition,
            )

            message = {
                "id": message_id,
                "sender": user_id,
                "receiver": receiver,
                "content": content,
                "sent_timestamp": sent_timestamp,
                "delivered_timestamp": None,
                "session_id": session_id,
                "experimental_condition": experimental_condition,
            }

            delivered = await manager.send_to_user(receiver, {"type": "message", **message})
            if delivered:
                delivered_timestamp = utc_now_iso()
                mark_delivered(session_id, message_id, delivered_timestamp)
                message["delivered_timestamp"] = delivered_timestamp

            await websocket.send_json({"type": "ack", "delivered": delivered, **message})
    except WebSocketDisconnect:
        manager.disconnect(user_id)
