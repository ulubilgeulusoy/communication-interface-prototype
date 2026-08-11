from __future__ import annotations

import hashlib


CONDITIONS = ("control", "delay_notice")


def assign_condition(session_id: str) -> str:
    """Deterministically assign a session to an experimental condition."""
    if not session_id:
        return CONDITIONS[0]

    digest = hashlib.sha256(session_id.encode("utf-8")).digest()
    return CONDITIONS[digest[0] % len(CONDITIONS)]
