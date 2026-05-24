"""SQLite-backed session memory for analyst Q&A history."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    query_id   TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    ts         INTEGER NOT NULL,
    question   TEXT NOT NULL,
    answer_json TEXT NOT NULL,
    feedback   TEXT
);
"""


class SessionStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def ensure_session(self, session_id: str | None) -> str:
        sid = session_id or str(uuid.uuid4())
        self._conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, created_at) VALUES (?, ?)",
            (sid, int(time.time())),
        )
        self._conn.commit()
        return sid

    def record(self, session_id: str, question: str, answer: dict) -> str:
        qid = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO turns (query_id, session_id, ts, question, answer_json) VALUES (?, ?, ?, ?, ?)",
            (qid, session_id, int(time.time()), question, json.dumps(answer, default=str)),
        )
        self._conn.commit()
        return qid

    def recent(self, session_id: str, limit: int = 5) -> list[dict]:
        rows = self._conn.execute(
            "SELECT query_id, ts, question, answer_json FROM turns "
            "WHERE session_id = ? ORDER BY ts DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [
            {"query_id": r[0], "ts": r[1], "question": r[2], "answer": json.loads(r[3])}
            for r in rows
        ]

    def feedback(self, query_id: str, value: str) -> None:
        self._conn.execute(
            "UPDATE turns SET feedback = ? WHERE query_id = ?", (value, query_id)
        )
        self._conn.commit()
