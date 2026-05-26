"""Local structured observability for agent runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage


DEFAULT_RUN_LOG_DIR = Path("logs/runs")


class JsonlRunObserver:
    """Append structured run events to one JSONL file."""

    def __init__(self, log_dir: Path = DEFAULT_RUN_LOG_DIR, run_id: str | None = None) -> None:
        self.run_id = run_id or uuid4().hex
        self.path = log_dir / f"{self.run_id}.jsonl"
        self._started_at = perf_counter()
        self._pending_tool_calls: dict[str, float] = {}

    def run_started(self, question: str, model: str, max_iterations: int) -> None:
        self.emit(
            "run.started",
            input_chars=len(question),
            model=model,
            max_iterations=max_iterations,
        )

    def router_completed(
        self,
        route: str,
        reason: str,
        latency_ms: int,
        token_usage: dict[str, int] | None,
    ) -> None:
        self.emit(
            "router.completed",
            route=route,
            reason=reason,
            latency_ms=latency_ms,
            token_usage=token_usage,
        )

    def router_failed(self, error: Exception, latency_ms: int) -> None:
        fields = _model_response_fields(getattr(error, "model_response", None))
        token_usage = getattr(error, "token_usage", None)
        response_metadata = getattr(error, "response_metadata", None)
        self.emit(
            "router.failed",
            stage="router",
            error_type=error.__class__.__name__,
            error_message=str(error),
            latency_ms=latency_ms,
            token_usage=token_usage,
            response_metadata=_json_safe_metadata(response_metadata),
            **fields,
        )

    def tool_called(self, name: str, tool_call_id: str | None, args: dict[str, Any]) -> None:
        if tool_call_id:
            self._pending_tool_calls[tool_call_id] = perf_counter()
        self.emit("tool.called", tool=name, tool_call_id=tool_call_id, args=args)

    def tool_completed(self, name: str | None, tool_call_id: str | None, content: Any) -> None:
        started_at = self._pending_tool_calls.pop(tool_call_id, None) if tool_call_id else None
        latency_ms = _elapsed_ms(started_at) if started_at is not None else None
        self.emit(
            "tool.completed",
            tool=name,
            tool_call_id=tool_call_id,
            ok=_tool_result_ok(content),
            error_type=_tool_error_type(content),
            result_bytes=_result_size_bytes(content),
            latency_ms=latency_ms,
        )

    def model_completed(self, message: BaseMessage) -> None:
        token_usage = _token_usage_from_message(message)
        if token_usage is not None:
            self.emit("model.completed", token_usage=token_usage)

    def model_failed(
        self,
        stage: str,
        error: Exception,
        latency_ms: int,
        model_response: str | None = None,
    ) -> None:
        self.emit(
            "model.failed",
            stage=stage,
            error_type=error.__class__.__name__,
            error_message=str(error),
            latency_ms=latency_ms,
            **_model_response_fields(model_response),
        )

    def run_completed(self, final_answer: str, status: str = "ok") -> None:
        self.emit(
            "run.completed",
            status=status,
            final_answer_chars=len(final_answer),
            latency_ms=_elapsed_ms(self._started_at),
        )

    def emit(self, event: str, **fields: Any) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event": event,
            **fields,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _elapsed_ms(started_at: float) -> int:
    return int((perf_counter() - started_at) * 1000)


def _result_size_bytes(value: Any) -> int:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return len(text.encode("utf-8"))


def _tool_result_ok(content: Any) -> bool:
    payload = _json_object(content)
    return payload.get("ok") is not False if payload is not None else True


def _tool_error_type(content: Any) -> str | None:
    payload = _json_object(content)
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    error_type = error.get("type")
    return str(error_type) if error_type else None


def _json_object(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str):
        return None
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _token_usage_from_message(message: BaseMessage) -> dict[str, int] | None:
    if isinstance(message, AIMessage) and message.usage_metadata:
        return _numeric_token_usage(message.usage_metadata)

    response_metadata = getattr(message, "response_metadata", {}) or {}
    token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
    if isinstance(token_usage, dict):
        return _numeric_token_usage(token_usage)
    return None


def _numeric_token_usage(token_usage: dict[str, Any]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in token_usage.items()
        if isinstance(value, int | float)
    }


def _model_response_fields(model_response: str | None) -> dict[str, Any]:
    if model_response is None:
        return {}
    return {
        "model_response": model_response,
        "model_response_chars": len(model_response),
    }


def _json_safe_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if metadata is None:
        return None
    return {
        str(key): value
        for key, value in metadata.items()
        if isinstance(value, str | int | float | bool) or value is None
    }
