from __future__ import annotations

import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from bitext_agent.graph import BitextAgent
from bitext_agent.config import AgentSettings
from bitext_agent.observability import JsonlRunObserver
from bitext_agent.profile import ProfileFact
from bitext_agent.router import RouterError


class RecursingAgent:
    def stream(self, *_args, **_kwargs):
        yield {
            "messages": [
                HumanMessage(content="How many refund requests?"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "filter_dataset",
                            "args": {"intent": "get_refund"},
                            "id": "tool-call-1",
                        }
                    ],
                ),
                ToolMessage(
                    content='{"filter_id": "abc123", "total_matches": 42}',
                    name="filter_dataset",
                    tool_call_id="tool-call-1",
                ),
            ]
        }
        raise GraphRecursionError("recursion limit reached")


class IncrementalRecursingAgent:
    def stream(self, *_args, **_kwargs):
        yield {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "filter_dataset",
                            "args": {"category": "REFUND"},
                            "id": "tool-call-1",
                        }
                    ],
                ),
                ToolMessage(
                    content='{"filter_id": "refund-filter", "total_matches": 3312}',
                    name="filter_dataset",
                    tool_call_id="tool-call-1",
                ),
            ]
        }
        yield {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "show_examples",
                            "args": {"filter_id": "refund-filter", "n": 3},
                            "id": "tool-call-2",
                        }
                    ],
                ),
                ToolMessage(
                    content='{"examples": [{"intent": "get_refund"}]}',
                    name="show_examples",
                    tool_call_id="tool-call-2",
                ),
            ]
        }
        raise GraphRecursionError("recursion limit reached")


class RepeatedToolRecursingAgent:
    def stream(self, *_args, **_kwargs):
        yield {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "show_examples",
                            "args": {"category": "REFUND", "n": 1},
                            "id": "tool-call-1",
                        }
                    ],
                ),
                ToolMessage(
                    content='{"examples": [{"intent": "get_refund"}]}',
                    name="show_examples",
                    tool_call_id="tool-call-1",
                ),
            ]
        }
        yield {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "show_examples",
                            "args": {"category": "ORDER", "n": 1},
                            "id": "tool-call-2",
                        }
                    ],
                ),
                ToolMessage(
                    content='{"examples": [{"intent": "track_order"}]}',
                    name="show_examples",
                    tool_call_id="tool-call-2",
                ),
            ]
        }
        raise GraphRecursionError("recursion limit reached")


class FailingRouterGraph:
    def invoke(self, *_args, **_kwargs):
        raise RouterError("router response invalid")


class StreamingErrorGraph:
    def __init__(self, error):
        self.error = error

    def stream(self, *_args, **_kwargs):
        raise self.error
        yield


class StreamingGraph:
    def __init__(self, chunks, final_state):
        self._chunks = chunks
        self._final_state = final_state

    def stream(self, *_args, **_kwargs):
        yield from self._chunks

    def get_state(self, *_args, **_kwargs):
        return SimpleNamespace(values=self._final_state)


class StreamingRouterThenErrorGraph:
    def __init__(self, router_update, error):
        self._router_update = router_update
        self._error = error

    def stream(self, *_args, **_kwargs):
        yield {"router": self._router_update}
        raise self._error


class ProfileRepositoryStub:
    def __init__(self, facts=None):
        self.facts = facts or []

    def list_facts(self, _user_id):
        return self.facts


class FailingProfileModel:
    def invoke(self, _messages):
        raise RuntimeError("profile extraction provider unavailable")


def test_agent_node_preserves_trace_when_inner_agent_hits_iteration_limit() -> None:
    agent = BitextAgent.__new__(BitextAgent)
    agent.settings = SimpleNamespace(max_iterations=12)
    agent.agent = RecursingAgent()

    result = agent._agent_node(
        {"messages": [HumanMessage(content="How many refund requests?")]}
    )

    assert any(isinstance(message, ToolMessage) for message in result["messages"])
    assert result["final_answer"].startswith("I could not complete")


def test_bitext_agent_configures_router_with_enough_output_tokens(monkeypatch, tmp_path) -> None:
    settings = AgentSettings(
        api_key="test-key",
        base_url="https://example.com/v1/",
        model_name="test-model",
        dataset_path="data/test.csv",
        max_iterations=12,
        checkpoint_path=tmp_path / "checkpoints.sqlite",
        profile_path=tmp_path / "profiles.sqlite",
    )
    model_calls = []

    def fake_make_chat_model(_settings, temperature, max_tokens, model_name=None):
        model_calls.append(
            {"temperature": temperature, "max_tokens": max_tokens, "model_name": model_name}
        )
        return SimpleNamespace()

    monkeypatch.setattr("bitext_agent.graph._make_chat_model", fake_make_chat_model)
    monkeypatch.setattr("bitext_agent.graph.create_agent", lambda **_kwargs: SimpleNamespace())

    BitextAgent(settings, SimpleNamespace())

    assert model_calls[0] == {"temperature": 0.0, "max_tokens": 10000, "model_name": None}
    assert model_calls[2] == {
        "temperature": 0.0,
        "max_tokens": 1024,
        "model_name": "Qwen/Qwen3.5-397B-A17B",
    }


def test_agent_node_preserves_all_incremental_tool_trace_on_iteration_limit() -> None:
    agent = BitextAgent.__new__(BitextAgent)
    agent.settings = SimpleNamespace(max_iterations=12)
    agent.agent = IncrementalRecursingAgent()

    result = agent._agent_node(
        {"messages": [HumanMessage(content="Show refund examples and then summarize them.")]}
    )

    tool_names = [
        message.name for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert tool_names == ["filter_dataset", "show_examples"]
    assert result["final_answer"].startswith("I could not complete")


def test_agent_node_preserves_repeated_tool_calls_with_different_ids() -> None:
    agent = BitextAgent.__new__(BitextAgent)
    agent.settings = SimpleNamespace(max_iterations=12)
    agent.agent = RepeatedToolRecursingAgent()

    result = agent._agent_node(
        {"messages": [HumanMessage(content="Show refund and order examples.")]}
    )

    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert [message.name for message in tool_messages] == ["show_examples", "show_examples"]
    assert [message.tool_call_id for message in tool_messages] == ["tool-call-1", "tool-call-2"]


def test_agent_node_emits_live_trace_events_for_tool_flow() -> None:
    agent = BitextAgent.__new__(BitextAgent)
    agent.settings = SimpleNamespace(max_iterations=12)
    agent.agent = IncrementalRecursingAgent()
    trace_events = []

    result = agent._agent_node(
        {"messages": [HumanMessage(content="Show refund examples and then summarize them.")]},
        trace_callback=trace_events.append,
    )

    assert [event["type"] for event in trace_events] == [
        "tool_call",
        "observation",
        "tool_call",
        "observation",
    ]
    assert trace_events[0]["name"] == "filter_dataset"
    assert trace_events[2]["name"] == "show_examples"
    assert result["final_answer"].startswith("I could not complete")


def test_ask_live_returns_technical_problem_and_emits_error_trace_on_router_failure() -> None:
    agent = BitextAgent.__new__(BitextAgent)
    agent.settings = SimpleNamespace(model_name="test-model", max_iterations=12)
    agent.graph = StreamingErrorGraph(RouterError("router response invalid"))
    trace_events = []

    result = agent.ask_live("How many refund requests?", trace_events.append)

    assert result["final_answer"] == (
        "I ran into an unexpected technical problem. "
        "Please try again in a few seconds or try a different query."
    )
    assert result["query_type"] == "error"
    assert trace_events == [
        {
            "type": "error",
            "stage": "router",
            "message": "router response invalid",
        }
    ]


def test_stream_yields_router_and_final_events() -> None:
    agent = BitextAgent.__new__(BitextAgent)
    agent.settings = SimpleNamespace(model_name="test-model", max_iterations=12)
    router_update = {
        "query_type": "out_of_scope",
        "route_reason": "Not a dataset question.",
        "token_usage": None,
    }
    agent.graph = StreamingGraph(
        [{"router": router_update}, {"decline": {"final_answer": "I can only answer dataset questions."}}],
        {
            **router_update,
            "final_answer": "I can only answer dataset questions.",
        },
    )

    events = list(agent.stream("Write a poem."))

    assert [event["type"] for event in events] == ["router", "final"]
    assert events[0]["query_type"] == "out_of_scope"
    assert events[-1]["final_answer"] == "I can only answer dataset questions."
    assert events[-1]["state"]["query_type"] == "out_of_scope"


def test_stream_yields_error_and_final_events_on_router_failure() -> None:
    agent = BitextAgent.__new__(BitextAgent)
    agent.settings = SimpleNamespace(model_name="test-model", max_iterations=12)
    agent.graph = StreamingErrorGraph(RouterError("router response invalid"))

    events = list(agent.stream("How many refund requests?"))

    assert events[0] == {
        "type": "error",
        "stage": "router",
        "message": "router response invalid",
    }
    assert events[-1]["type"] == "final"
    assert events[-1]["state"]["query_type"] == "error"


def test_ask_live_emits_router_failure_metadata_for_empty_length_response() -> None:
    agent = BitextAgent.__new__(BitextAgent)
    agent.settings = SimpleNamespace(model_name="test-model", max_iterations=12)
    agent.graph = StreamingErrorGraph(
        RouterError(
            "Router response did not contain JSON.",
            model_response="",
            token_usage={"output_tokens": 256},
            response_metadata={"finish_reason": "length"},
        )
    )
    trace_events = []

    agent.ask_live("What is the dataset called?", trace_events.append)

    assert trace_events == [
        {
            "type": "error",
            "stage": "router",
            "message": "Router response did not contain JSON.",
            "finish_reason": "length",
            "token_usage": {"output_tokens": 256},
            "model_response_chars": 0,
        }
    ]


def test_ask_returns_technical_problem_on_router_failure() -> None:
    agent = BitextAgent.__new__(BitextAgent)
    agent.settings = SimpleNamespace(max_iterations=12)
    agent.graph = FailingRouterGraph()

    result = agent.ask("How many refund requests?")

    assert result["final_answer"] == (
        "I ran into an unexpected technical problem. "
        "Please try again in a few seconds or try a different query."
    )
    assert result["query_type"] == "error"
    assert result["route_reason"] == "router response invalid"


def test_ask_live_writes_run_and_router_observability_events(tmp_path) -> None:
    agent = BitextAgent.__new__(BitextAgent)
    agent.settings = SimpleNamespace(model_name="test-model", max_iterations=12)
    router_update = {
        "query_type": "out_of_scope",
        "route_reason": "Not a dataset question.",
        "token_usage": {"input_tokens": 20, "output_tokens": 4},
    }
    agent.graph = StreamingGraph(
        [{"router": router_update}, {"decline": {"final_answer": "I can only answer dataset questions."}}],
        {
            **router_update,
            "final_answer": "I can only answer dataset questions.",
        },
    )
    observer = JsonlRunObserver(log_dir=tmp_path, run_id="run-123")

    result = agent.ask_live("Write a poem.", lambda _event: None, observer=observer)

    events = [
        json.loads(line)
        for line in observer.path.read_text(encoding="utf-8").splitlines()
    ]
    assert result["run_id"] == "run-123"
    assert result["observability_path"] == str(observer.path)
    assert [event["event"] for event in events] == [
        "run.started",
        "router.completed",
        "run.completed",
    ]
    assert events[1]["route"] == "out_of_scope"
    assert events[1]["token_usage"] == {"input_tokens": 20, "output_tokens": 4}


def test_agent_node_writes_tool_observability_events(tmp_path) -> None:
    agent = BitextAgent.__new__(BitextAgent)
    agent.settings = SimpleNamespace(max_iterations=12)
    agent.agent = IncrementalRecursingAgent()
    observer = JsonlRunObserver(log_dir=tmp_path, run_id="run-123")

    agent._agent_node(
        {"messages": [HumanMessage(content="Show refund examples and then summarize them.")]},
        observer=observer,
    )

    events = [
        json.loads(line)
        for line in observer.path.read_text(encoding="utf-8").splitlines()
    ]
    event_names = [event["event"] for event in events]
    assert event_names == [
        "tool.called",
        "tool.completed",
        "tool.called",
        "tool.completed",
    ]
    assert events[0]["tool"] == "filter_dataset"
    assert events[0]["args"] == {"category": "REFUND"}
    assert events[1]["result_bytes"] > 0


def test_ask_live_logs_full_agent_response_on_agent_failure(tmp_path) -> None:
    agent = BitextAgent.__new__(BitextAgent)
    agent.settings = SimpleNamespace(model_name="test-model", max_iterations=12)
    router_update = {
        "query_type": "structured",
        "route_reason": "Dataset question.",
        "token_usage": None,
    }
    error = RuntimeError("agent model returned invalid output")
    error.response = SimpleNamespace(text="raw agent response")
    agent.graph = StreamingRouterThenErrorGraph(router_update, error)
    observer = JsonlRunObserver(log_dir=tmp_path, run_id="run-123")
    trace_events = []

    result = agent.ask_live("How many refunds?", trace_events.append, observer=observer)

    events = [
        json.loads(line)
        for line in observer.path.read_text(encoding="utf-8").splitlines()
    ]
    assert result["final_answer"] == (
        "I ran into an unexpected technical problem. "
        "Please try again in a few seconds or try a different query."
    )
    assert trace_events[-1] == {
        "type": "error",
        "stage": "agent",
        "message": "agent model returned invalid output",
    }
    assert events[2]["event"] == "model.failed"
    assert events[2]["stage"] == "agent"
    assert events[2]["model_response"] == "raw agent response"


def test_profile_info_node_returns_saved_profile_facts() -> None:
    agent = BitextAgent.__new__(BitextAgent)
    agent.settings = SimpleNamespace(user_id="alex")
    agent.profile_repository = ProfileRepositoryStub(
        [ProfileFact(fact_id="fact-1", fact="You prefer refund analytics.")]
    )

    result = agent._profile_info_node({"messages": [HumanMessage(content="What do you remember?")]})

    assert result["final_answer"] == "I remember:\n- You prefer refund analytics."
    assert result["proposed_profile_facts"] == []


def test_profile_extractor_node_skips_non_dataset_routes() -> None:
    agent = BitextAgent.__new__(BitextAgent)

    result = agent._profile_extractor_node(
        {
            "query_type": "profile_info",
            "messages": [HumanMessage(content="What do you remember about me?")],
        }
    )

    assert result == {"proposed_profile_facts": []}


def test_profile_extractor_node_uses_all_user_messages_and_final_answer(monkeypatch) -> None:
    agent = BitextAgent.__new__(BitextAgent)
    agent.settings = SimpleNamespace(user_id="alex")
    agent.profile_model = SimpleNamespace()
    agent.profile_repository = ProfileRepositoryStub(
        [ProfileFact(fact_id="fact-1", fact="You prefer examples.")]
    )
    captured = {}

    def fake_extract(model, *, existing_facts, user_messages, final_answer):
        captured["model"] = model
        captured["existing_facts"] = existing_facts
        captured["user_messages"] = user_messages
        captured["final_answer"] = final_answer
        return ["You prefer refund analytics."]

    monkeypatch.setattr("bitext_agent.graph.extract_profile_facts", fake_extract)

    result = agent._profile_extractor_node(
        {
            "query_type": "structured",
            "messages": [
                HumanMessage(content="My name is Alex."),
                AIMessage(content="Hello."),
                HumanMessage(content="Show refund examples."),
            ],
            "final_answer": "Here are examples.",
        }
    )

    assert result == {"proposed_profile_facts": ["You prefer refund analytics."]}
    assert captured["user_messages"] == ["My name is Alex.", "Show refund examples."]
    assert captured["final_answer"] == "Here are examples."
    assert captured["existing_facts"][0].fact == "You prefer examples."


def test_profile_extractor_node_preserves_answer_when_profile_model_fails() -> None:
    agent = BitextAgent.__new__(BitextAgent)
    agent.settings = SimpleNamespace(user_id="alex")
    agent.profile_model = FailingProfileModel()
    agent.profile_repository = ProfileRepositoryStub()

    result = agent._profile_extractor_node(
        {
            "query_type": "structured",
            "messages": [HumanMessage(content="Show refund examples.")],
            "final_answer": "Here are refund examples.",
        }
    )

    assert result == {"proposed_profile_facts": []}
