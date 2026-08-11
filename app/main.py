from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .db import init_db, list_messages, log_message, mark_delivered, utc_now_iso
from .experiment import assign_condition


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "frontend"

app = FastAPI(title="Two-User Communication Prototype")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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
    return {
        "session_id": session_id,
        "experimental_condition": assign_condition(session_id),
    }


@app.get("/api/messages")
def get_messages(session_id: str | None = Query(default=None)) -> list[dict]:
    return list_messages(session_id)


@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str) -> None:
    await manager.connect(user_id, websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)

            receiver = str(data["receiver"])
            content = str(data["content"])
            session_id = str(data["session_id"])
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
                mark_delivered(message_id, delivered_timestamp)
                message["delivered_timestamp"] = delivered_timestamp

            await websocket.send_json({"type": "ack", "delivered": delivered, **message})
    except WebSocketDisconnect:
        manager.disconnect(user_id)
