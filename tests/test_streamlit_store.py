from __future__ import annotations

import pytest

from bitext_agent.streamlit_store import StreamlitSessionStore, UnknownUiSessionError


def test_store_lists_sessions_by_most_recent_turn(tmp_path) -> None:
    store = StreamlitSessionStore(tmp_path / "streamlit_sessions.sqlite")
    store.create_session("session-first")
    store.create_session("session-second")

    store.add_turn("session-first", "user", "Show refund counts.")

    sessions = store.list_sessions()

    assert [session.session_id for session in sessions] == [
        "session-first",
        "session-second",
    ]


def test_store_persists_turns_and_trace_events(tmp_path) -> None:
    path = tmp_path / "streamlit_sessions.sqlite"
    store = StreamlitSessionStore(path)
    store.create_session("session-refund")
    trace_events = [
        {"type": "router", "query_type": "structured"},
        {"type": "tool_call", "name": "count_rows", "args": {"category": "REFUND"}},
    ]

    store.add_turn("session-refund", "user", "How many refund requests?")
    store.add_turn(
        "session-refund",
        "assistant",
        "There are 2992 refund rows.",
        trace_events,
    )
    store.close()

    reopened_store = StreamlitSessionStore(path)
    turns = reopened_store.list_turns("session-refund")

    assert [(turn.role, turn.content) for turn in turns] == [
        ("user", "How many refund requests?"),
        ("assistant", "There are 2992 refund rows."),
    ]
    assert turns[1].trace_events == trace_events


def test_store_rejects_unknown_session(tmp_path) -> None:
    store = StreamlitSessionStore(tmp_path / "streamlit_sessions.sqlite")

    with pytest.raises(UnknownUiSessionError):
        store.require_session("session-missing")

    with pytest.raises(UnknownUiSessionError):
        store.add_turn("session-missing", "user", "Show categories.")


def test_store_sets_first_question_as_session_title(tmp_path) -> None:
    store = StreamlitSessionStore(tmp_path / "streamlit_sessions.sqlite")
    store.create_session("session-title")

    store.update_title_from_question(
        "session-title",
        "Show the distribution of refund-related intents in the dataset.",
    )
    store.update_title_from_question("session-title", "This should not replace the title.")

    session = store.require_session("session-title")
    assert session.title == "Show the distribution of refund-related intents in the dataset."
