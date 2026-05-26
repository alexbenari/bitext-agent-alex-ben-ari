"""Query routing for the Bitext agent graph."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError

from bitext_agent.error_details import extract_model_response
from bitext_agent.prompts import ROUTER_PROMPT


QueryType = Literal["structured", "unstructured", "profile_info", "out_of_scope"]


class RouteDecision(BaseModel):
    """Router output for one user question."""

    model_config = ConfigDict(extra="forbid")

    query_type: QueryType = Field(description="Question routing class.")
    reason: str = Field(description="Short reason for the routing decision.")
    _token_usage: dict[str, int] | None = PrivateAttr(default=None)

    @property
    def token_usage(self) -> dict[str, int] | None:
        """Token usage metadata from the routing model response."""

        return self._token_usage


class RouterError(RuntimeError):
    """Raised when the routing model cannot produce a valid route."""

    def __init__(
        self,
        message: str,
        model_response: str | None = None,
        token_usage: dict[str, int] | None = None,
        response_metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.model_response = model_response
        self.token_usage = token_usage
        self.response_metadata = response_metadata


def route_query(model, question: str) -> RouteDecision:
    """Classify a user question with the routing model."""

    from langchain_core.messages import HumanMessage, SystemMessage

    try:
        response = model.invoke(
            [
                SystemMessage(content=ROUTER_PROMPT),
                HumanMessage(content=question),
            ]
        )
    except Exception as error:
        raise RouterError(
            f"router model call failed: {error}",
            model_response=extract_model_response(error),
        ) from error

    text = str(response.content)
    token_usage = _extract_token_usage(response)
    response_metadata = _extract_response_metadata(response)
    try:
        decision = RouteDecision.model_validate_json(_extract_json(text))
    except (ValidationError, json.JSONDecodeError) as error:
        raise RouterError(
            f"invalid route response: {error}",
            model_response=text,
            token_usage=token_usage,
            response_metadata=response_metadata,
        ) from error
    except ValueError as error:
        raise RouterError(
            str(error),
            model_response=text,
            token_usage=token_usage,
            response_metadata=response_metadata,
        ) from error
    decision._token_usage = token_usage
    return decision


def _extract_json(text: str) -> str:
    candidate = _strip_markdown_code_fence(text).lstrip()
    start = candidate.find("{")
    if start == -1:
        raise ValueError("Router response did not contain JSON.")

    decoder = json.JSONDecoder()
    try:
        _, end = decoder.raw_decode(candidate[start:])
    except json.JSONDecodeError as error:
        raise ValueError(f"Router response did not contain valid JSON: {error}") from error
    return candidate[start : start + end]


def _strip_markdown_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _extract_token_usage(response) -> dict[str, int] | None:
    usage_metadata = getattr(response, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        return _numeric_token_usage(usage_metadata)

    response_metadata = getattr(response, "response_metadata", {}) or {}
    token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
    if isinstance(token_usage, dict):
        return _numeric_token_usage(token_usage)
    return None


def _extract_response_metadata(response) -> dict[str, Any] | None:
    response_metadata = getattr(response, "response_metadata", None)
    return dict(response_metadata) if isinstance(response_metadata, dict) else None


def _numeric_token_usage(token_usage: dict) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in token_usage.items()
        if isinstance(value, int | float)
    }
