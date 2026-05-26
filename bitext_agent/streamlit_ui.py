"""Helpers shared by the Streamlit UI."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from uuid import uuid4


def new_session_id() -> str:
    """Return a new conversation session id."""

    return f"session-{uuid4()}"


def collect_stream_result(events: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Split an agent event stream into final state and trace events."""

    final_state: dict[str, Any] = {}
    trace_events: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") == "final":
            final_state = event.get("state", {})
        else:
            trace_events.append(event)
    return final_state, trace_events


def trace_event_label(event: dict[str, Any]) -> str:
    """Return a compact label for one trace event."""

    event_type = event.get("type")
    if event_type == "router":
        return f"router: {event.get('query_type', 'unknown')}"
    if event_type == "tool_call":
        return f"tool call: {event.get('name', 'unknown')}"
    if event_type == "observation":
        return f"observation: {event.get('name', 'unknown')}"
    if event_type == "error":
        return f"error: {event.get('stage', 'unknown')}"
    return str(event_type or "event")


def trace_event_counts(trace_events: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Count trace event types for compact run summaries."""

    counts = {"router": 0, "tool_call": 0, "observation": 0, "error": 0}
    for event in trace_events:
        event_type = event.get("type")
        if event_type in counts:
            counts[event_type] += 1
    return counts
