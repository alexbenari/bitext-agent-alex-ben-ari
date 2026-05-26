"""SQLite-backed session metadata for the Streamlit UI."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_STREAMLIT_SESSION_DB = Path("data/streamlit_sessions.sqlite")


class UnknownUiSessionError(RuntimeError):
    """Raised when the UI is asked to switch to an unknown session."""


@dataclass(frozen=True)
class UiSession:
    """Stored metadata for one visible Streamlit chat session."""

    session_id: str
    created_at: str
    last_used_at: str
    title: str


@dataclass(frozen=True)
class UiTurn:
    """One persisted chat turn and its trace events for the Streamlit UI."""

    role: str
    content: str
    trace_events: list[dict[str, Any]]
    created_at: str


class StreamlitSessionStore:
    """Persist Streamlit-visible sessions and chat turns."""

    def __init__(self, path: Path = DEFAULT_STREAMLIT_SESSION_DB) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._ensure_schema()

    def create_session(self, session_id: str, title: str = "New session") -> UiSession:
        now = _utc_now()
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO sessions (session_id, created_at, last_used_at, title)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO NOTHING
                """,
                (session_id, now, now, title),
            )
        session = self.get_session(session_id)
        if session is None:
            raise RuntimeError(f"Could not create Streamlit session: {session_id}")
        return session

    def get_session(self, session_id: str) -> UiSession | None:
        row = self._connection.execute(
            """
            SELECT session_id, created_at, last_used_at, title
            FROM sessions
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        return _session_from_row(row) if row is not None else None

    def require_session(self, session_id: str) -> UiSession:
        session = self.get_session(session_id)
        if session is None:
            raise UnknownUiSessionError(f"Unknown Streamlit session id: {session_id}")
        return session

    def list_sessions(self) -> list[UiSession]:
        rows = self._connection.execute(
            """
            SELECT
                sessions.session_id,
                sessions.created_at,
                sessions.last_used_at,
                sessions.title,
                MAX(turns.id) AS last_turn_id
            FROM sessions
            LEFT JOIN turns ON turns.session_id = sessions.session_id
            GROUP BY sessions.session_id
            ORDER BY
                last_turn_id IS NOT NULL DESC,
                sessions.last_used_at DESC,
                last_turn_id DESC,
                sessions.created_at DESC
            """
        ).fetchall()
        return [_session_from_row(row) for row in rows]

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        trace_events: list[dict[str, Any]] | None = None,
    ) -> UiTurn:
        self.require_session(session_id)
        now = _utc_now()
        trace_events = trace_events or []
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO turns (session_id, role, content, trace_events_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    content,
                    json.dumps(trace_events, ensure_ascii=False),
                    now,
                ),
            )
            self._connection.execute(
                """
                UPDATE sessions
                SET last_used_at = ?
                WHERE session_id = ?
                """,
                (now, session_id),
            )
        return UiTurn(role=role, content=content, trace_events=trace_events, created_at=now)

    def list_turns(self, session_id: str) -> list[UiTurn]:
        self.require_session(session_id)
        rows = self._connection.execute(
            """
            SELECT role, content, trace_events_json, created_at
            FROM turns
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()
        return [_turn_from_row(row) for row in rows]

    def update_title_from_question(self, session_id: str, question: str) -> None:
        session = self.require_session(session_id)
        if session.title != "New session":
            return
        title = _question_title(question)
        with self._connection:
            self._connection.execute(
                """
                UPDATE sessions
                SET title = ?
                WHERE session_id = ?
                """,
                (title, session_id),
            )

    def close(self) -> None:
        self._connection.close()

    def _ensure_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    last_used_at TEXT NOT NULL,
                    title TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    trace_events_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
                """
            )


def _session_from_row(row: sqlite3.Row) -> UiSession:
    return UiSession(
        session_id=row["session_id"],
        created_at=row["created_at"],
        last_used_at=row["last_used_at"],
        title=row["title"],
    )


def _turn_from_row(row: sqlite3.Row) -> UiTurn:
    trace_events_json = row["trace_events_json"] or "[]"
    trace_events = json.loads(trace_events_json)
    return UiTurn(
        role=row["role"],
        content=row["content"],
        trace_events=trace_events if isinstance(trace_events, list) else [],
        created_at=row["created_at"],
    )


def _question_title(question: str) -> str:
    compact_question = " ".join(question.split())
    if len(compact_question) <= 64:
        return compact_question or "New session"
    return f"{compact_question[:61]}..."


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")
