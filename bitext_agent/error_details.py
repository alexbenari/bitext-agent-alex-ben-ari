"""Helpers for extracting diagnostic details from provider errors."""

from __future__ import annotations

from typing import Any


def extract_model_response(error: Exception) -> str | None:
    """Return the raw provider response carried by an exception, when available."""

    for value in _candidate_values(error):
        text = _stringify_response_value(value)
        if text:
            return text
    return None


def _candidate_values(error: Exception) -> list[Any]:
    response = getattr(error, "response", None)
    values = [response] if response is not None else []
    values.extend(
        getattr(error, attr, None)
        for attr in ("body", "response_body", "content", "text")
    )
    if response is not None:
        values.extend(
            getattr(response, attr, None)
            for attr in ("text", "content", "body")
        )
    return values


def _stringify_response_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return None
