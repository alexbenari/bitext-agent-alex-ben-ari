from __future__ import annotations

from bitext_agent.streamlit_ui import (
    collect_stream_result,
    new_session_id,
    trace_event_counts,
    trace_event_label,
)


def test_new_session_id_returns_session_prefixed_unique_id() -> None:
    first_session_id = new_session_id()
    second_session_id = new_session_id()

    assert first_session_id.startswith("session-")
    assert second_session_id.startswith("session-")
    assert first_session_id != second_session_id


def test_collect_stream_result_splits_trace_events_from_final_state() -> None:
    router_event = {
        "type": "router",
        "query_type": "structured",
        "reason": "Dataset count question.",
    }
    tool_event = {
        "type": "tool_call",
        "name": "count_rows",
        "args": {"category": "REFUND"},
    }
    final_state = {
        "query_type": "structured",
        "final_answer": "There are 2992 refund rows.",
    }

    state, trace_events = collect_stream_result(
        [
            router_event,
            tool_event,
            {"type": "final", "state": final_state},
        ]
    )

    assert state == final_state
    assert trace_events == [router_event, tool_event]


def test_trace_event_label_describes_known_event_types() -> None:
    assert trace_event_label({"type": "router", "query_type": "structured"}) == (
        "router: structured"
    )
    assert trace_event_label({"type": "tool_call", "name": "count_rows"}) == (
        "tool call: count_rows"
    )
    assert trace_event_label({"type": "observation", "name": "count_rows"}) == (
        "observation: count_rows"
    )
    assert trace_event_label({"type": "error", "stage": "router"}) == "error: router"


def test_trace_event_label_handles_unknown_event_shape() -> None:
    assert trace_event_label({"type": "custom"}) == "custom"
    assert trace_event_label({}) == "event"


def test_trace_event_counts_returns_known_event_type_counts() -> None:
    counts = trace_event_counts(
        [
            {"type": "router"},
            {"type": "tool_call"},
            {"type": "tool_call"},
            {"type": "observation"},
            {"type": "custom"},
        ]
    )

    assert counts == {
        "router": 1,
        "tool_call": 2,
        "observation": 1,
        "error": 0,
    }
