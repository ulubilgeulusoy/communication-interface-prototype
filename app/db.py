from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
SESSIONS_DIR = BASE_DIR / "sessions"
SESSION_ID_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")


def sanitize_session_id(session_id: str) -> str:
    cleaned = SESSION_ID_PATTERN.sub("-", str(session_id).strip())
    cleaned = cleaned.strip("-")
    if not cleaned:
        raise ValueError("Session ID must contain at least one letter or number.")
    return cleaned


def get_db_path(session_id: str) -> Path:
    safe_session_id = sanitize_session_id(session_id)
    return SESSIONS_DIR / f"{safe_session_id}.sqlite3"


def ensure_sessions_dir() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def get_connection(session_id: str) -> sqlite3.Connection:
    ensure_sessions_dir()
    conn = sqlite3.connect(get_db_path(session_id))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(session_id: str | None = None) -> None:
    ensure_sessions_dir()
    if session_id is None:
        return

    with get_connection(session_id) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender TEXT NOT NULL,
                receiver TEXT NOT NULL,
                content TEXT NOT NULL,
                sent_timestamp TEXT NOT NULL,
                delivered_timestamp TEXT,
                session_id TEXT NOT NULL,
                experimental_condition TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                model TEXT NOT NULL,
                input_text TEXT NOT NULL,
                output_text TEXT NOT NULL
            )
            """
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_message(
    *,
    sender: str,
    receiver: str,
    content: str,
    sent_timestamp: str,
    session_id: str,
    experimental_condition: str,
) -> int:
    init_db(session_id)
    with get_connection(session_id) as conn:
        cursor = conn.execute(
            """
            INSERT INTO messages (
                sender,
                receiver,
                content,
                sent_timestamp,
                session_id,
                experimental_condition
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                sender,
                receiver,
                content,
                sent_timestamp,
                session_id,
                experimental_condition,
            ),
        )
        return int(cursor.lastrowid)


def mark_delivered(session_id: str, message_id: int, delivered_timestamp: str) -> None:
    init_db(session_id)
    with get_connection(session_id) as conn:
        conn.execute(
            "UPDATE messages SET delivered_timestamp = ? WHERE id = ?",
            (delivered_timestamp, message_id),
        )


def list_messages(session_id: str) -> list[dict[str, Any]]:
    init_db(session_id)
    with get_connection(session_id) as conn:
        rows = conn.execute("SELECT * FROM messages ORDER BY id ASC").fetchall()
        return [dict(row) for row in rows]


def list_recent_messages(session_id: str, limit: int = 12) -> list[dict[str, Any]]:
    init_db(session_id)
    with get_connection(session_id) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM messages
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in reversed(rows)]


def log_llm_interaction(
    *,
    timestamp: str,
    session_id: str,
    user_id: str,
    model: str,
    input_text: str,
    output_text: str,
) -> int:
    init_db(session_id)
    with get_connection(session_id) as conn:
        cursor = conn.execute(
            """
            INSERT INTO llm_interactions (
                timestamp,
                session_id,
                user_id,
                model,
                input_text,
                output_text
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (timestamp, session_id, user_id, model, input_text, output_text),
        )
        return int(cursor.lastrowid)


def list_llm_interactions(session_id: str) -> list[dict[str, Any]]:
    init_db(session_id)
    with get_connection(session_id) as conn:
        rows = conn.execute("SELECT * FROM llm_interactions ORDER BY id ASC").fetchall()
        return [dict(row) for row in rows]
