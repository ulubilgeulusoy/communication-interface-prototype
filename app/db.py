from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DB_PATH = Path(__file__).resolve().parent.parent / "messages.sqlite3"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
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
    with get_connection() as conn:
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


def mark_delivered(message_id: int, delivered_timestamp: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE messages SET delivered_timestamp = ? WHERE id = ?",
            (delivered_timestamp, message_id),
        )


def list_messages(session_id: str | None = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM messages"
    params: tuple[str, ...] = ()
    if session_id:
        sql += " WHERE session_id = ?"
        params = (session_id,)
    sql += " ORDER BY id ASC"

    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]


def list_recent_messages(session_id: str, limit: int = 12) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
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
    with get_connection() as conn:
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
