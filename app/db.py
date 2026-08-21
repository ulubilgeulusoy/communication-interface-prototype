from __future__ import annotations

import re
import sqlite3
import json
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
            CREATE TABLE IF NOT EXISTS session_settings (
                session_id TEXT PRIMARY KEY,
                delay_ms INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender TEXT NOT NULL,
                receiver TEXT NOT NULL,
                content TEXT NOT NULL,
                attachments_json TEXT NOT NULL DEFAULT '[]',
                sent_timestamp TEXT NOT NULL,
                delivered_timestamp TEXT,
                delay_ms INTEGER NOT NULL DEFAULT 0,
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
                input_attachments_json TEXT NOT NULL DEFAULT '[]',
                output_text TEXT NOT NULL,
                retrieved_sources_json TEXT,
                retrieved_chunks_json TEXT,
                selected_models_json TEXT NOT NULL DEFAULT '[]',
                raw_vision_findings TEXT,
                vision_error TEXT,
                response_latency_ms INTEGER
            )
            """
        )
        _ensure_column(
            conn,
            table_name="messages",
            column_name="attachments_json",
            column_definition="TEXT NOT NULL DEFAULT '[]'",
        )
        _ensure_column(
            conn,
            table_name="messages",
            column_name="delay_ms",
            column_definition="INTEGER NOT NULL DEFAULT 0",
        )
        _ensure_column(
            conn,
            table_name="llm_interactions",
            column_name="input_attachments_json",
            column_definition="TEXT NOT NULL DEFAULT '[]'",
        )
        _ensure_column(
            conn,
            table_name="llm_interactions",
            column_name="retrieved_sources_json",
            column_definition="TEXT",
        )
        _ensure_column(
            conn,
            table_name="llm_interactions",
            column_name="retrieved_chunks_json",
            column_definition="TEXT",
        )
        _ensure_column(
            conn,
            table_name="llm_interactions",
            column_name="selected_models_json",
            column_definition="TEXT NOT NULL DEFAULT '[]'",
        )
        _ensure_column(
            conn,
            table_name="llm_interactions",
            column_name="raw_vision_findings",
            column_definition="TEXT",
        )
        _ensure_column(
            conn,
            table_name="llm_interactions",
            column_name="vision_error",
            column_definition="TEXT",
        )
        _ensure_column(
            conn,
            table_name="llm_interactions",
            column_name="response_latency_ms",
            column_definition="INTEGER",
        )
        conn.execute(
            """
            INSERT INTO session_settings (session_id, delay_ms)
            VALUES (?, 0)
            ON CONFLICT(session_id) DO NOTHING
            """,
            (session_id,),
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_message(
    *,
    sender: str,
    receiver: str,
    content: str,
    attachments: list[dict[str, Any]] | None,
    sent_timestamp: str,
    delay_ms: int,
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
                attachments_json,
                sent_timestamp,
                delay_ms,
                session_id,
                experimental_condition
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sender,
                receiver,
                content,
                json.dumps(attachments or []),
                sent_timestamp,
                max(0, int(delay_ms)),
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


def get_session_delay_ms(session_id: str) -> int:
    init_db(session_id)
    with get_connection(session_id) as conn:
        row = conn.execute(
            "SELECT delay_ms FROM session_settings WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return 0
        return max(0, int(row["delay_ms"]))


def set_session_delay_ms(session_id: str, delay_ms: int) -> int:
    safe_delay_ms = max(0, int(delay_ms))
    init_db(session_id)
    with get_connection(session_id) as conn:
        conn.execute(
            """
            INSERT INTO session_settings (session_id, delay_ms)
            VALUES (?, ?)
            ON CONFLICT(session_id) DO UPDATE SET delay_ms = excluded.delay_ms
            """,
            (session_id, safe_delay_ms),
        )
    return safe_delay_ms


def list_messages(session_id: str) -> list[dict[str, Any]]:
    init_db(session_id)
    with get_connection(session_id) as conn:
        rows = conn.execute("SELECT * FROM messages ORDER BY id ASC").fetchall()
        return [_row_to_message(row) for row in rows]


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
        return [_row_to_message(row) for row in reversed(rows)]


def log_llm_interaction(
    *,
    timestamp: str,
    session_id: str,
    user_id: str,
    model: str,
    input_text: str,
    input_attachments: list[dict[str, Any]] | None = None,
    output_text: str,
    retrieved_sources_json: str | None = None,
    retrieved_chunks_json: str | None = None,
    selected_models: list[str] | None = None,
    raw_vision_findings: str | None = None,
    vision_error: str | None = None,
    response_latency_ms: int | None = None,
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
                input_attachments_json,
                output_text,
                retrieved_sources_json,
                retrieved_chunks_json,
                selected_models_json,
                raw_vision_findings,
                vision_error,
                response_latency_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                session_id,
                user_id,
                model,
                input_text,
                json.dumps(input_attachments or []),
                output_text,
                retrieved_sources_json,
                retrieved_chunks_json,
                json.dumps(selected_models or []),
                raw_vision_findings,
                vision_error,
                response_latency_ms,
            ),
        )
        return int(cursor.lastrowid)


def update_llm_interaction_latency(
    *,
    session_id: str,
    interaction_id: int,
    response_latency_ms: int,
) -> None:
    init_db(session_id)
    with get_connection(session_id) as conn:
        conn.execute(
            """
            UPDATE llm_interactions
            SET response_latency_ms = ?
            WHERE id = ? AND session_id = ?
            """,
            (max(0, int(response_latency_ms)), interaction_id, session_id),
        )


def list_llm_interactions(session_id: str) -> list[dict[str, Any]]:
    init_db(session_id)
    with get_connection(session_id) as conn:
        rows = conn.execute("SELECT * FROM llm_interactions ORDER BY id ASC").fetchall()
        return [_row_to_llm_interaction(row) for row in rows]


def list_recent_llm_interactions(
    session_id: str,
    *,
    user_id: str,
    limit: int = 6,
) -> list[dict[str, Any]]:
    init_db(session_id)
    with get_connection(session_id) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM llm_interactions
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        return [_row_to_llm_interaction(row) for row in reversed(rows)]


def _ensure_column(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    column_name: str,
    column_definition: str,
) -> None:
    existing_columns = {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name in existing_columns:
        return

    conn.execute(
        f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
    )


def _row_to_message(row: sqlite3.Row) -> dict[str, Any]:
    message = dict(row)
    try:
        parsed_attachments = json.loads(str(message.get("attachments_json") or "[]"))
    except (TypeError, ValueError):
        parsed_attachments = []

    message["attachments"] = [
        item for item in parsed_attachments if isinstance(item, dict)
    ]
    return message


def _row_to_llm_interaction(row: sqlite3.Row) -> dict[str, Any]:
    interaction = dict(row)
    try:
        parsed_attachments = json.loads(
            str(interaction.get("input_attachments_json") or "[]")
        )
    except (TypeError, ValueError):
        parsed_attachments = []

    interaction["input_attachments"] = [
        item for item in parsed_attachments if isinstance(item, dict)
    ]
    return interaction
