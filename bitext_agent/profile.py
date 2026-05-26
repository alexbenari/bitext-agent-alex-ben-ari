"""Persistent user profile storage and profile fact extraction helpers."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bitext_agent.prompts import PROFILE_EXTRACTOR_PROMPT


PROFILE_EXTRACTION_MODEL = "Qwen/Qwen3.5-397B-A17B"


@dataclass(frozen=True)
class ProfileFact:
    """One persisted fact about a user."""

    fact_id: str
    fact: str


class ProposedProfileFacts(BaseModel):
    """Structured output from the profile extraction model."""

    model_config = ConfigDict(extra="forbid")

    facts: list[str] = Field(default_factory=list)


class ProfileRepository:
    """SQLite-backed storage for user profile facts."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.db_path)
        self._connection.row_factory = sqlite3.Row
        self.setup()

    def setup(self) -> None:
        """Create profile tables when they do not already exist."""

        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                display_name TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS user_profile_facts (
                fact_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                fact TEXT NOT NULL,
                source_session_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, fact),
                FOREIGN KEY(user_id) REFERENCES users(user_id)
            );
            """
        )
        self._connection.commit()

    def ensure_user(self, user_id: str) -> None:
        """Create the user row if it does not already exist."""

        now = _utc_now()
        self._connection.execute(
            """
            INSERT INTO users (user_id, display_name, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET updated_at = excluded.updated_at
            """,
            (user_id, user_id, now, now),
        )
        self._connection.commit()

    def list_facts(self, user_id: str) -> list[ProfileFact]:
        """Return saved profile facts for a user."""

        self.ensure_user(user_id)
        rows = self._connection.execute(
            """
            SELECT fact_id, fact
            FROM user_profile_facts
            WHERE user_id = ?
            ORDER BY created_at, fact
            """,
            (user_id,),
        ).fetchall()
        return [ProfileFact(fact_id=row["fact_id"], fact=row["fact"]) for row in rows]

    def add_facts(
        self,
        user_id: str,
        facts: list[str],
        *,
        source_session_id: str,
    ) -> int:
        """Persist approved profile facts and return the number of inserted rows."""

        self.ensure_user(user_id)
        now = _utc_now()
        inserted = 0
        for fact in _normalized_facts(facts):
            cursor = self._connection.execute(
                """
                INSERT OR IGNORE INTO user_profile_facts
                    (fact_id, user_id, fact, source_session_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (str(uuid4()), user_id, fact, source_session_id, now, now),
            )
            inserted += cursor.rowcount
        self._connection.commit()
        return inserted

    def close(self) -> None:
        """Close the SQLite connection."""

        self._connection.close()


def format_profile_answer(facts: list[ProfileFact]) -> str:
    """Format profile facts for a user-facing answer."""

    if not facts:
        return "I do not have any saved profile facts for you yet."
    lines = ["I remember:"]
    lines.extend(f"- {fact.fact}" for fact in facts)
    return "\n".join(lines)


def extract_profile_facts(
    model,
    *,
    existing_facts: list[ProfileFact],
    user_messages: list[str],
    final_answer: str,
) -> list[str]:
    """Ask an LLM to propose durable user profile facts."""

    payload = {
        "existing_profile": [fact.fact for fact in existing_facts],
        "user_messages": user_messages,
        "final_agent_answer": final_answer,
    }
    response = model.invoke(
        [
            SystemMessage(content=PROFILE_EXTRACTOR_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]
    )
    try:
        parsed = ProposedProfileFacts.model_validate_json(_extract_json(str(response.content)))
    except (ValidationError, ValueError, json.JSONDecodeError):
        return []
    existing = {fact.fact.casefold() for fact in existing_facts}
    return [fact for fact in _normalized_facts(parsed.facts) if fact.casefold() not in existing]


def _normalized_facts(facts: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for fact in facts:
        clean = " ".join(str(fact).split())
        if not clean or clean.casefold() in seen:
            continue
        normalized.append(clean)
        seen.add(clean.casefold())
    return normalized


def _extract_json(text: str) -> str:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            candidate = "\n".join(lines[1:-1]).strip()
    start = candidate.find("{")
    if start == -1:
        raise ValueError("Profile extractor response did not contain JSON.")
    decoder = json.JSONDecoder()
    _, end = decoder.raw_decode(candidate[start:])
    return candidate[start : start + end]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
