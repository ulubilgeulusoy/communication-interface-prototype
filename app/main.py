from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .db import (
    init_db,
    list_llm_interactions,
    list_messages,
    list_recent_messages,
    log_llm_interaction,
    log_message,
    mark_delivered,
    sanitize_session_id,
    utc_now_iso,
)
from .experiment import assign_condition
from .llm_service import DEFAULT_MODEL, OllamaService, OllamaServiceError


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "frontend"

app = FastAPI(title="Two-User Communication Prototype")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
ollama_service = OllamaService()
LLM_HISTORY_LIMIT = 12


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


class SessionHistory(BaseModel):
    session_id: str
    messages: list[dict]
    llm_interactions: list[dict]


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

    try:
        reply = await ollama_service.generate_reply(
            message_text=payload.message_text,
            conversation_history=conversation_history,
            model=payload.model,
        )
    except OllamaServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    log_llm_interaction(
        timestamp=timestamp,
        session_id=safe_session_id,
        user_id=payload.user_id,
        model=reply.model or payload.model or DEFAULT_MODEL,
        input_text=payload.message_text,
        output_text=reply.output_text,
    )

    return LLMReply(
        session_id=safe_session_id,
        user_id=payload.user_id,
        model=reply.model,
        output_text=reply.output_text,
        timestamp=timestamp,
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
