from __future__ import annotations

import asyncio
import json
from functools import lru_cache
from pathlib import Path
import re
from dataclasses import dataclass
from uuid import uuid4
from mimetypes import guess_type

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .db import (
    get_session_delay_ms,
    init_db,
    list_llm_interactions,
    list_messages,
    list_recent_llm_interactions,
    list_recent_messages,
    log_llm_interaction,
    log_message,
    mark_delivered,
    sanitize_session_id,
    set_session_delay_ms,
    update_llm_interaction_latency,
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
UPLOADS_DIR = BASE_DIR / "uploads"

app = FastAPI(title="Two-User Communication Prototype")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
ollama_service = OllamaService()
rag_service = RAGService(ollama_service)
LLM_HISTORY_LIMIT = 12
LLM_THREAD_HISTORY_LIMIT = 4


class LLMRequest(BaseModel):
    session_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    message_text: str = Field(min_length=1)
    model: str | None = None


class LLMReply(BaseModel):
    interaction_id: int
    session_id: str
    user_id: str
    model: str
    output_text: str
    timestamp: str
    response_latency_ms: int | None = None
    retrieved_chunks: list[dict] = Field(default_factory=list)
    used_retrieval: bool = False
    retrieval_mode: str = "global"
    active_document_title: str = ""
    active_document_id: str = ""
    retrieval_query: str = ""
    retrieval_requested_top_k: int = 0
    retrieval_raw_candidate_count: int = 0
    retrieval_filtered_candidate_count: int = 0
    retrieval_returned_chunk_count: int = 0
    max_similarity_distance: float | None = None


class SessionHistory(BaseModel):
    session_id: str
    messages: list[dict]
    llm_interactions: list[dict]
    delay_ms: int = 0


class UserMessageReply(BaseModel):
    id: int
    sender: str
    receiver: str
    content: str
    attachments: list[dict] = Field(default_factory=list)
    sent_timestamp: str
    delivered_timestamp: str | None = None
    session_id: str
    experimental_condition: str
    delay_ms: int = 0
    delivered: bool = False
    delivery_pending: bool = False


class DelaySettings(BaseModel):
    session_id: str = Field(min_length=1)
    delay_ms: int = Field(default=0, ge=0, le=600000)


class LLMLatencyUpdate(BaseModel):
    session_id: str = Field(min_length=1)
    interaction_id: int = Field(ge=1)
    response_latency_ms: int = Field(ge=0, le=3600000)


class RAGReindexReply(BaseModel):
    indexed_files: int
    indexed_chunks: int
    collection_name: str
    embedding_model: str
    knowledge_base_path: str


class LLMContextReply(BaseModel):
    session_id: str
    user_id: str
    retrieval_mode: str
    active_document_title: str
    active_document_id: str


QUOTED_TITLE_PATTERN = re.compile(r'"([^"]{3,})"|\'([^\']{3,})\'')
GUIDE_ID_PATTERN = re.compile(r"\b1\d{5}\b")
SWITCH_DOCUMENT_PATTERN = re.compile(
    r"\b(switch to|use|talk about|focus on|now use|different procedure|new document)\b",
    re.IGNORECASE,
)
CLEAR_DOCUMENT_PHRASES = (
    "clear document",
    "clear the document",
    "clear current document",
    "clear the current document",
    "discard document",
    "discard the document",
    "discard current document",
    "discard the current document",
    "discard documentation",
    "discard the documentation",
    "discard current documentation",
    "discard the current documentation",
    "remove document",
    "remove the document",
    "remove current document",
    "remove the current document",
    "remove documentation",
    "remove the documentation",
    "remove manual",
    "remove the manual",
    "ignore document",
    "ignore documents",
    "ignore the document",
    "ignore the documents",
    "ignore documentation",
    "ignore the documentation",
    "stop using the document",
    "stop using document",
    "stop using the documentation",
    "stop using documentation",
    "stop using the manual",
    "stop using manual",
    "use general knowledge",
    "general knowledge only",
    "no manual",
)


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    title: str
    filename: str
    guide_id: str


@dataclass
class LLMContextState:
    retrieval_mode: str = "global"
    active_document_title: str = ""
    active_document_id: str = ""


def ensure_uploads_dir() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def build_attachment_url(session_id: str, attachment_id: str) -> str:
    return f"/api/attachments/{session_id}/{attachment_id}"


def store_user_attachment(
    *,
    session_id: str,
    upload: UploadFile,
) -> dict:
    ensure_uploads_dir()
    session_dir = UPLOADS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    original_name = Path(upload.filename or "attachment").name or "attachment"
    attachment_id = uuid4().hex
    suffix = Path(original_name).suffix
    stored_name = f"{attachment_id}{suffix}"
    stored_path = session_dir / stored_name
    payload = upload.file.read()
    stored_path.write_bytes(payload)

    detected_type = upload.content_type or guess_type(original_name)[0] or "application/octet-stream"
    return {
        "id": attachment_id,
        "name": original_name,
        "stored_name": stored_name,
        "content_type": detected_type,
        "size_bytes": stored_path.stat().st_size,
        "url": build_attachment_url(session_id, attachment_id),
    }


def resolve_attachment_path(session_id: str, attachment_id: str) -> Path:
    session_dir = UPLOADS_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Attachment not found")

    for candidate in session_dir.iterdir():
        if candidate.is_file() and candidate.name.startswith(attachment_id):
            return candidate

    raise HTTPException(status_code=404, detail="Attachment not found")


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


def should_clear_document_context(message_text: str) -> bool:
    normalized = " ".join(str(message_text).lower().split())
    return any(phrase in normalized for phrase in CLEAR_DOCUMENT_PHRASES)


@lru_cache(maxsize=1)
def get_document_catalog() -> tuple[DocumentRecord, ...]:
    records: list[DocumentRecord] = []
    for document in load_documents(KNOWLEDGE_BASE_PATH):
        title = _extract_document_title(document.text) or Path(document.filename).stem.replace("_", " ").replace("-", " ").strip()
        guide_id_match = GUIDE_ID_PATTERN.search(document.text) or GUIDE_ID_PATTERN.search(document.filename)
        records.append(
            DocumentRecord(
                document_id=document.document_id,
                title=title.strip(),
                filename=document.filename,
                guide_id=guide_id_match.group(0) if guide_id_match else "",
            )
        )
    return tuple(records)


def _extract_document_title(text: str) -> str:
    for line in str(text).splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned
    return ""


def resolve_document_reference(message_text: str) -> DocumentRecord | None:
    quoted_title = _extract_quoted_title(message_text)
    guide_id_match = GUIDE_ID_PATTERN.search(message_text)
    lowered_message = message_text.lower()

    catalog = get_document_catalog()
    if guide_id_match:
        guide_id = guide_id_match.group(0)
        for record in catalog:
            if record.guide_id == guide_id:
                return record

    if quoted_title:
        lowered_title = quoted_title.lower().strip()
        exact_matches = [record for record in catalog if record.title.lower() == lowered_title]
        if exact_matches:
            return exact_matches[0]

        partial_matches = [
            record
            for record in catalog
            if lowered_title in record.title.lower() or lowered_title in record.filename.lower()
        ]
        if partial_matches:
            return partial_matches[0]

    switch_requested = bool(SWITCH_DOCUMENT_PATTERN.search(message_text))
    if switch_requested:
        for record in catalog:
            if record.title.lower() in lowered_message:
                return record

    return None


class LLMContextManager:
    def __init__(self) -> None:
        self._states: dict[tuple[str, str], LLMContextState] = {}

    def get(self, session_id: str, user_id: str) -> LLMContextState:
        return self._states.get((session_id, user_id), LLMContextState())

    def set(self, session_id: str, user_id: str, state: LLMContextState) -> None:
        self._states[(session_id, user_id)] = state

    def clear(self, session_id: str, user_id: str) -> None:
        self._states[(session_id, user_id)] = LLMContextState(
            retrieval_mode="global",
            active_document_title="",
            active_document_id="",
        )


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
llm_context_manager = LLMContextManager()


async def deliver_message_with_delay(
    *,
    sender: str,
    receiver: str,
    session_id: str,
    message_id: int,
    message: dict,
    delay_ms: int,
) -> None:
    if delay_ms > 0:
        await asyncio.sleep(delay_ms / 1000)

    delivered = await manager.send_to_user(receiver, {"type": "message", **message})
    if delivered:
        delivered_timestamp = utc_now_iso()
        mark_delivered(session_id, message_id, delivered_timestamp)
        message["delivered_timestamp"] = delivered_timestamp

    await manager.send_to_user(
        sender,
        {
            "type": "delivery_update",
            "id": message_id,
            "session_id": session_id,
            "delivered": delivered,
            "delivered_timestamp": message.get("delivered_timestamp"),
            "delay_ms": delay_ms,
        },
    )


@app.on_event("startup")
def startup() -> None:
    init_db()
    ensure_uploads_dir()


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
        delay_ms=get_session_delay_ms(safe_session_id),
    )


@app.get("/api/attachments/{session_id}/{attachment_id}")
def get_attachment(session_id: str, attachment_id: str) -> FileResponse:
    safe_session_id = sanitize_session_id(session_id)
    attachment_path = resolve_attachment_path(safe_session_id, attachment_id)
    return FileResponse(attachment_path)


@app.get("/api/session/{session_id}/delay", response_model=DelaySettings)
def get_session_delay(session_id: str) -> DelaySettings:
    safe_session_id = sanitize_session_id(session_id)
    return DelaySettings(
        session_id=safe_session_id,
        delay_ms=get_session_delay_ms(safe_session_id),
    )


@app.post("/api/session/delay", response_model=DelaySettings)
def update_session_delay(payload: DelaySettings) -> DelaySettings:
    safe_session_id = sanitize_session_id(payload.session_id)
    return DelaySettings(
        session_id=safe_session_id,
        delay_ms=set_session_delay_ms(safe_session_id, payload.delay_ms),
    )


@app.post("/api/messages/user", response_model=UserMessageReply)
async def send_user_message(
    session_id: str = Form(...),
    sender: str = Form(...),
    receiver: str = Form(...),
    content: str = Form(""),
    experimental_condition: str = Form(""),
    attachments: list[UploadFile] | None = File(None),
) -> UserMessageReply:
    safe_session_id = sanitize_session_id(session_id)
    clean_content = str(content).strip()
    safe_sender = str(sender).strip()
    safe_receiver = str(receiver).strip()
    if not safe_sender or not safe_receiver:
        raise HTTPException(status_code=400, detail="Sender and receiver are required")

    stored_attachments = [
        store_user_attachment(session_id=safe_session_id, upload=attachment)
        for attachment in (attachments or [])
        if attachment.filename
    ]
    if not clean_content and not stored_attachments:
        raise HTTPException(status_code=400, detail="Message content or an attachment is required")

    resolved_condition = str(experimental_condition or assign_condition(safe_session_id))
    delay_ms = get_session_delay_ms(safe_session_id)
    sent_timestamp = utc_now_iso()
    message_id = log_message(
        sender=safe_sender,
        receiver=safe_receiver,
        content=clean_content,
        attachments=stored_attachments,
        sent_timestamp=sent_timestamp,
        delay_ms=delay_ms,
        session_id=safe_session_id,
        experimental_condition=resolved_condition,
    )
    message = {
        "id": message_id,
        "sender": safe_sender,
        "receiver": safe_receiver,
        "content": clean_content,
        "attachments": stored_attachments,
        "sent_timestamp": sent_timestamp,
        "delivered_timestamp": None,
        "session_id": safe_session_id,
        "experimental_condition": resolved_condition,
        "delay_ms": delay_ms,
    }

    asyncio.create_task(
        deliver_message_with_delay(
            sender=safe_sender,
            receiver=safe_receiver,
            session_id=safe_session_id,
            message_id=message_id,
            message=dict(message),
            delay_ms=delay_ms,
        )
    )

    return UserMessageReply(
        delivered=False,
        delivery_pending=True,
        **message,
    )


@app.get("/api/llm/context", response_model=LLMContextReply)
def get_llm_context(
    session_id: str = Query(..., min_length=1),
    user_id: str = Query(..., min_length=1),
) -> LLMContextReply:
    safe_session_id = sanitize_session_id(session_id)
    state = llm_context_manager.get(safe_session_id, user_id)
    if not state.active_document_title and not state.active_document_id:
        inferred_title, inferred_document_id = infer_active_document_context(
            safe_session_id,
            user_id=user_id,
        )
        state = LLMContextState(
            retrieval_mode="document" if inferred_document_id else state.retrieval_mode,
            active_document_title=inferred_title,
            active_document_id=inferred_document_id,
        )
        llm_context_manager.set(safe_session_id, user_id, state)

    return LLMContextReply(
        session_id=safe_session_id,
        user_id=user_id,
        retrieval_mode=state.retrieval_mode,
        active_document_title=state.active_document_title,
        active_document_id=state.active_document_id,
    )


@app.post("/api/llm/context/clear", response_model=LLMContextReply)
def clear_llm_context(payload: LLMRequest) -> LLMContextReply:
    safe_session_id = sanitize_session_id(payload.session_id)
    llm_context_manager.clear(safe_session_id, payload.user_id)
    state = llm_context_manager.get(safe_session_id, payload.user_id)
    return LLMContextReply(
        session_id=safe_session_id,
        user_id=payload.user_id,
        retrieval_mode=state.retrieval_mode,
        active_document_title=state.active_document_title,
        active_document_id=state.active_document_id,
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
    current_state = llm_context_manager.get(safe_session_id, payload.user_id)
    explicit_clear = should_clear_document_context(payload.message_text)
    explicit_document = resolve_document_reference(payload.message_text)

    if explicit_clear:
        current_state = LLMContextState(
            retrieval_mode="disabled",
            active_document_title="",
            active_document_id="",
        )
    elif explicit_document:
        current_state = LLMContextState(
            retrieval_mode="document",
            active_document_title=explicit_document.title,
            active_document_id=explicit_document.document_id,
        )
    elif not current_state.active_document_title and not current_state.active_document_id:
        inferred_title, inferred_document_id = infer_active_document_context(
            safe_session_id,
            user_id=payload.user_id,
        )
        current_state = LLMContextState(
            retrieval_mode="document" if inferred_document_id else "global",
            active_document_title=inferred_title,
            active_document_id=inferred_document_id,
        )

    llm_context_manager.set(safe_session_id, payload.user_id, current_state)

    try:
        rag_reply = await rag_service.generate_reply(
            message_text=payload.message_text,
            session_id=safe_session_id,
            conversation_history=conversation_history,
            llm_thread_history=llm_thread_history,
            active_document_hint=current_state.active_document_title,
            active_document_id=current_state.active_document_id,
            retrieval_mode=current_state.retrieval_mode,
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

    interaction_id = log_llm_interaction(
        timestamp=timestamp,
        session_id=safe_session_id,
        user_id=payload.user_id,
        model=rag_reply.llm_response.model or payload.model or DEFAULT_MODEL,
        input_text=payload.message_text,
        output_text=rag_reply.llm_response.output_text,
        retrieved_sources_json=json.dumps(retrieved_chunks_payload),
    )

    if rag_reply.used_retrieval and retrieved_chunks_payload and current_state.retrieval_mode != "disabled":
        first_chunk = rag_reply.retrieved_chunks[0]
        resolved_title = current_state.active_document_title or Path(first_chunk.filename).stem.replace("_", " ").replace("-", " ").strip()
        current_state = LLMContextState(
            retrieval_mode="document",
            active_document_title=resolved_title,
            active_document_id=first_chunk.document_id,
        )
    elif current_state.retrieval_mode == "disabled":
        current_state = LLMContextState(
            retrieval_mode="disabled",
            active_document_title="",
            active_document_id="",
        )
    elif explicit_clear:
        current_state = LLMContextState(
            retrieval_mode="disabled",
            active_document_title="",
            active_document_id="",
        )

    llm_context_manager.set(safe_session_id, payload.user_id, current_state)

    return LLMReply(
        interaction_id=interaction_id,
        session_id=safe_session_id,
        user_id=payload.user_id,
        model=rag_reply.llm_response.model,
        output_text=rag_reply.llm_response.output_text,
        timestamp=timestamp,
        response_latency_ms=None,
        retrieved_chunks=retrieved_chunks_payload,
        used_retrieval=rag_reply.used_retrieval,
        retrieval_mode=current_state.retrieval_mode,
        active_document_title=current_state.active_document_title,
        active_document_id=current_state.active_document_id,
        retrieval_query=(
            rag_reply.retrieval_diagnostics.query if rag_reply.retrieval_diagnostics else ""
        ),
        retrieval_requested_top_k=(
            rag_reply.retrieval_diagnostics.requested_top_k if rag_reply.retrieval_diagnostics else 0
        ),
        retrieval_raw_candidate_count=(
            rag_reply.retrieval_diagnostics.raw_candidate_count if rag_reply.retrieval_diagnostics else 0
        ),
        retrieval_filtered_candidate_count=(
            rag_reply.retrieval_diagnostics.filtered_candidate_count
            if rag_reply.retrieval_diagnostics
            else 0
        ),
        retrieval_returned_chunk_count=(
            rag_reply.retrieval_diagnostics.returned_chunk_count
            if rag_reply.retrieval_diagnostics
            else 0
        ),
        max_similarity_distance=(
            rag_reply.retrieval_diagnostics.max_similarity_distance
            if rag_reply.retrieval_diagnostics
            else None
        ),
    )


@app.post("/api/llm/latency")
def update_llm_latency(payload: LLMLatencyUpdate) -> dict[str, int | str]:
    safe_session_id = sanitize_session_id(payload.session_id)
    update_llm_interaction_latency(
        session_id=safe_session_id,
        interaction_id=payload.interaction_id,
        response_latency_ms=payload.response_latency_ms,
    )
    return {
        "session_id": safe_session_id,
        "interaction_id": payload.interaction_id,
        "response_latency_ms": payload.response_latency_ms,
    }


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
    get_document_catalog.cache_clear()
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
            delay_ms = get_session_delay_ms(session_id)
            sent_timestamp = utc_now_iso()

            message_id = log_message(
                sender=user_id,
                receiver=receiver,
                content=content,
                attachments=[],
                sent_timestamp=sent_timestamp,
                delay_ms=delay_ms,
                session_id=session_id,
                experimental_condition=experimental_condition,
            )

            message = {
                "id": message_id,
                "sender": user_id,
                "receiver": receiver,
                "content": content,
                "attachments": [],
                "sent_timestamp": sent_timestamp,
                "delivered_timestamp": None,
                "session_id": session_id,
                "experimental_condition": experimental_condition,
                "delay_ms": delay_ms,
            }

            asyncio.create_task(
                deliver_message_with_delay(
                    sender=user_id,
                    receiver=receiver,
                    session_id=session_id,
                    message_id=message_id,
                    message=dict(message),
                    delay_ms=delay_ms,
                )
            )

            await websocket.send_json(
                {
                    "type": "ack",
                    "delivered": False,
                    "delivery_pending": True,
                    **message,
                }
            )
    except WebSocketDisconnect:
        manager.disconnect(user_id)
