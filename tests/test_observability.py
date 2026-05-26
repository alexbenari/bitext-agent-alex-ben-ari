from __future__ import annotations

import json

from langchain_core.messages import AIMessage

from bitext_agent.observability import JsonlRunObserver
from bitext_agent.router import RouterError


def read_events(observer: JsonlRunObserver) -> list[dict]:
    return [
        json.loads(line)
        for line in observer.path.read_text(encoding="utf-8").splitlines()
    ]


def test_jsonl_run_observer_writes_one_event_per_line(tmp_path) -> None:
    observer = JsonlRunObserver(log_dir=tmp_path, run_id="run-123")

    observer.run_started("How many refunds?", model="test-model", max_iterations=12)
    observer.router_completed(
        route="structured",
        reason="Dataset count question.",
        latency_ms=15,
        token_usage={"input_tokens": 10, "output_tokens": 3},
    )
    observer.run_completed("There are 3 refund rows.")

    events = read_events(observer)
    assert [event["event"] for event in events] == [
        "run.started",
        "router.completed",
        "run.completed",
    ]
    assert {event["run_id"] for event in events} == {"run-123"}
    assert events[1]["route"] == "structured"
    assert events[1]["token_usage"] == {"input_tokens": 10, "output_tokens": 3}


def test_jsonl_run_observer_records_tool_result_size_and_error_type(tmp_path) -> None:
    observer = JsonlRunObserver(log_dir=tmp_path, run_id="run-123")

    observer.tool_called("count_rows", "tool-call-1", {"filter_id": "missing"})
    observer.tool_completed(
        "count_rows",
        "tool-call-1",
        '{"ok": false, "error": {"type": "recoverable_tool_error"}}',
    )

    events = read_events(observer)
    assert events[0]["event"] == "tool.called"
    assert events[0]["args"] == {"filter_id": "missing"}
    assert events[1]["event"] == "tool.completed"
    assert events[1]["ok"] is False
    assert events[1]["error_type"] == "recoverable_tool_error"
    assert events[1]["result_bytes"] > 0
    assert isinstance(events[1]["latency_ms"], int)


def test_jsonl_run_observer_ignores_nested_token_usage_details(tmp_path) -> None:
    observer = JsonlRunObserver(log_dir=tmp_path, run_id="run-123")
    message = AIMessage(
        content="",
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": 3,
            "total_tokens": 13,
            "input_token_details": {"cache_read": 2},
        },
    )

    observer.model_completed(message)

    events = read_events(observer)
    assert events == [
        {
            "event": "model.completed",
            "run_id": "run-123",
            "token_usage": {
                "input_tokens": 10,
                "output_tokens": 3,
                "total_tokens": 13,
            },
            "ts": events[0]["ts"],
        }
    ]


def test_jsonl_run_observer_records_full_router_response_on_failure(tmp_path) -> None:
    observer = JsonlRunObserver(log_dir=tmp_path, run_id="run-123")
    error = RouterError(
        "Router response did not contain JSON.",
        model_response="plain text instead of json",
        token_usage={"output_tokens": 256},
        response_metadata={"finish_reason": "length"},
    )

    observer.router_failed(error, latency_ms=15)

    events = read_events(observer)
    assert events[0]["event"] == "router.failed"
    assert events[0]["model_response"] == "plain text instead of json"
    assert events[0]["token_usage"] == {"output_tokens": 256}
    assert events[0]["response_metadata"] == {"finish_reason": "length"}


def test_jsonl_run_observer_records_full_agent_response_on_failure(tmp_path) -> None:
    observer = JsonlRunObserver(log_dir=tmp_path, run_id="run-123")

    observer.model_failed(
        stage="agent",
        error=RuntimeError("invalid tool call"),
        latency_ms=15,
        model_response='{"malformed": "tool call"}',
    )

    events = read_events(observer)
    assert events[0]["event"] == "model.failed"
    assert events[0]["stage"] == "agent"
    assert events[0]["model_response"] == '{"malformed": "tool call"}'
