from __future__ import annotations

from types import SimpleNamespace

import pytest

from bitext_agent.router import RouterError, route_query


class StaticRouterModel:
    def __init__(
        self,
        content: str,
        usage_metadata: dict[str, int] | None = None,
        response_metadata: dict | None = None,
    ) -> None:
        self.content = content
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata

    def invoke(self, _messages):
        return SimpleNamespace(
            content=self.content,
            usage_metadata=self.usage_metadata,
            response_metadata=self.response_metadata,
        )


class FailingRouterModel:
    def invoke(self, _messages):
        raise RuntimeError("router provider unavailable")


def test_route_query_raises_router_error_for_invalid_model_json() -> None:
    with pytest.raises(RouterError, match="Router response did not contain JSON") as error:
        route_query(StaticRouterModel("not json"), "How many refund requests?")

    assert error.value.model_response == "not json"


def test_route_query_preserves_metadata_when_router_returns_empty_text() -> None:
    with pytest.raises(RouterError, match="Router response did not contain JSON") as error:
        route_query(
            StaticRouterModel(
                "",
                usage_metadata={"input_tokens": 395, "output_tokens": 256},
                response_metadata={"finish_reason": "length", "id": "chatcmpl-test"},
            ),
            "What is the dataset called?",
        )

    assert error.value.model_response == ""
    assert error.value.token_usage == {"input_tokens": 395, "output_tokens": 256}
    assert error.value.response_metadata == {
        "finish_reason": "length",
        "id": "chatcmpl-test",
    }


def test_route_query_raises_router_error_for_invalid_route_schema() -> None:
    with pytest.raises(RouterError, match="invalid route response"):
        route_query(
            StaticRouterModel('{"query_type": "maybe", "reason": "invalid"}'),
            "How many refund requests?",
        )


def test_route_query_raises_router_error_for_extra_json_keys() -> None:
    with pytest.raises(RouterError, match="invalid route response"):
        route_query(
            StaticRouterModel(
                '{"query_type": "structured", "reason": "valid", "extra": "unexpected"}'
            ),
            "How many refund requests?",
        )


def test_route_query_uses_first_json_object_when_multiple_objects_are_present() -> None:
    decision = route_query(
        StaticRouterModel(
            """```json
{"query_type": "structured", "reason": "first"}
```
{"query_type": "out_of_scope", "reason": "second"}"""
        ),
        "How many refund requests?",
    )

    assert decision.query_type == "structured"
    assert decision.reason == "first"


def test_route_query_raises_router_error_when_model_call_fails() -> None:
    with pytest.raises(RouterError, match="router model call failed"):
        route_query(FailingRouterModel(), "How many refund requests?")


def test_route_query_attaches_token_usage_from_model_response() -> None:
    decision = route_query(
        StaticRouterModel(
            '{"query_type": "structured", "reason": "Dataset question."}',
            usage_metadata={"input_tokens": 20, "output_tokens": 5},
        ),
        "How many refund requests?",
    )

    assert decision.token_usage == {"input_tokens": 20, "output_tokens": 5}


def test_route_query_accepts_profile_info_route() -> None:
    decision = route_query(
        StaticRouterModel(
            '{"query_type": "profile_info", "reason": "The user asks what is remembered."}'
        ),
        "What do you remember about me?",
    )

    assert decision.query_type == "profile_info"
