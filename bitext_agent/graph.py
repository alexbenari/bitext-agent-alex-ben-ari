"""LangGraph construction for the Bitext customer-service analyst."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import suppress
from time import perf_counter
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from bitext_agent.config import AgentSettings
from bitext_agent.dataset import BitextDataset
from bitext_agent.error_details import extract_model_response
from bitext_agent.observability import JsonlRunObserver
from bitext_agent.profile import (
    PROFILE_EXTRACTION_MODEL,
    ProfileRepository,
    extract_profile_facts,
    format_profile_answer,
)
from bitext_agent.prompts import AGENT_SYSTEM_PROMPT
from bitext_agent.router import RouteDecision, RouterError, route_query
from bitext_agent.tools import DatasetToolbox


class AgentState(TypedDict, total=False):
    """State shared by the router and ReAct agent nodes."""

    messages: Annotated[list[AnyMessage], add_messages]
    recent_filters: Annotated[dict[str, dict[str, Any]], _merge_recent_filters]
    query_type: str
    route_reason: str
    final_answer: str
    proposed_profile_facts: list[str]
    token_usage: dict[str, int] | None
    run_id: str
    observability_path: str


TraceCallback = Callable[[dict[str, Any]], None]
AgentEvent = dict[str, Any]
RunObserver = JsonlRunObserver
TECHNICAL_PROBLEM_MESSAGE = (
    "I ran into an unexpected technical problem. "
    "Please try again in a few seconds or try a different query."
)


class BitextAgent:
    """Runnable Bitext analyst graph with routing and max-iteration fallback."""

    def __init__(self, settings: AgentSettings, dataset: BitextDataset) -> None:
        self.settings = settings
        self.toolbox = DatasetToolbox(dataset)
        self.router_model = _make_chat_model(settings, temperature=0.0, max_tokens=10000)
        self.agent_model = _make_chat_model(settings, temperature=0.1, max_tokens=10000)
        self.profile_model = _make_chat_model(
            settings,
            temperature=0.0,
            max_tokens=1024,
            model_name=PROFILE_EXTRACTION_MODEL,
        )
        self.profile_repository = ProfileRepository(settings.profile_path)
        self.profile_repository.ensure_user(settings.user_id)
        self.agent = create_agent(
            model=self.agent_model,
            tools=self.toolbox.tools(),
            system_prompt=AGENT_SYSTEM_PROMPT,
        )
        self._active_trace_callback: TraceCallback | None = None
        self._active_observer: RunObserver | None = None
        self._checkpoint_connection = self._open_checkpoint_connection()
        self._checkpointer = SqliteSaver(self._checkpoint_connection)
        self._checkpointer.setup()
        self.graph = self._build_graph()

    def ask(self, question: str) -> AgentState:
        """Run one question through the graph and return the final state."""

        config = self._graph_config()
        try:
            return self.graph.invoke(
                {"messages": [HumanMessage(content=question)]},
                config=config,
            )
        except RouterError as error:
            return _technical_problem_state(question, error)
        except GraphRecursionError:
            return {
                "messages": [
                    HumanMessage(content=question),
                    AIMessage(content=_max_iteration_message()),
                ],
                "query_type": "fallback",
                "route_reason": "The graph hit the configured iteration limit.",
                "final_answer": _max_iteration_message(),
            }

    def ask_live(
        self,
        question: str,
        trace_callback: TraceCallback,
        observer: RunObserver | None = None,
    ) -> AgentState:
        """Run one question and emit trace events as they occur."""

        final_state: AgentState = {}
        for event in self.stream(question, observer=observer):
            if event.get("type") == "final":
                final_state = event["state"]
            else:
                trace_callback(event)
        return final_state

    def stream(
        self,
        question: str,
        observer: RunObserver | None = None,
    ) -> Iterator[AgentEvent]:
        """Run one question and yield router, tool, error, and final events."""

        config = self._graph_config()
        if observer is not None:
            observer.run_started(
                question,
                model=self.settings.model_name,
                max_iterations=self.settings.max_iterations,
            )

        final_state: AgentState = {}
        pending_events: list[AgentEvent] = []
        router_started_at = perf_counter()
        self._active_trace_callback = pending_events.append
        self._active_observer = observer
        try:
            for chunk in self.graph.stream(
                {"messages": [HumanMessage(content=question)]},
                config=config,
                stream_mode="updates",
            ):
                if "router" in chunk:
                    _emit_router_update(
                        chunk["router"],
                        pending_events.append,
                        observer,
                        latency_ms=_elapsed_ms(router_started_at),
                    )
                if (
                    "decline" in chunk
                    or "profile_info" in chunk
                    or "profile_extractor" in chunk
                ):
                    final_state = self.graph.get_state(config).values
                yield from _drain_events(pending_events)
            if not final_state:
                final_state = self.graph.get_state(config).values
            if observer is not None:
                observer.run_completed(final_state.get("final_answer", ""), status="ok")
            yield from _drain_events(pending_events)
            yield _final_event(_attach_observability_state(final_state, observer))
        except RouterError as error:
            result = self._handle_live_error(
                question,
                error,
                stage="router",
                trace_callback=pending_events.append,
                observer=observer,
            )
            yield from _drain_events(pending_events)
            yield _final_event(result)
        except Exception as error:
            result = self._handle_live_error(
                question,
                error,
                stage="agent",
                trace_callback=pending_events.append,
                observer=observer,
            )
            yield from _drain_events(pending_events)
            yield _final_event(result)
        finally:
            self._active_trace_callback = None
            self._active_observer = None

    def has_saved_session(self) -> bool:
        """Return whether the configured session id has persisted checkpoints."""

        return any(self.graph.get_state_history(self._graph_config()))

    def close(self) -> None:
        """Close the SQLite checkpoint connection held by this agent."""

        with suppress(sqlite3.Error):
            self._checkpoint_connection.close()
        if hasattr(self, "profile_repository"):
            self.profile_repository.close()

    def _handle_live_error(
        self,
        question: str,
        error: Exception,
        *,
        stage: str,
        trace_callback: TraceCallback,
        observer: RunObserver | None,
    ) -> AgentState:
        if observer is not None:
            if isinstance(error, RouterError):
                observer.router_failed(error, 0)
            else:
                observer.model_failed(
                    stage=stage,
                    error=error,
                    latency_ms=0,
                    model_response=extract_model_response(error),
                )
        trace_event = {"type": "error", "stage": stage, "message": str(error)}
        if isinstance(error, RouterError):
            trace_event.update(_router_error_trace_fields(error))
        trace_callback(trace_event)
        result = _technical_problem_state(question, error)
        if observer is not None:
            observer.run_completed(result["final_answer"], status="error")
        return _attach_observability_state(result, observer)

    def _open_checkpoint_connection(self) -> sqlite3.Connection:
        self.settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.settings.checkpoint_path, check_same_thread=False)

    def _graph_config(self) -> RunnableConfig:
        return {
            "configurable": {"thread_id": getattr(self.settings, "session_id", "default")},
            "recursion_limit": self.settings.max_iterations,
        }

    def _current_trace_callback(self, trace_callback: TraceCallback | None) -> TraceCallback | None:
        return trace_callback or getattr(self, "_active_trace_callback", None)

    def _current_observer(self, observer: RunObserver | None) -> RunObserver | None:
        return observer or getattr(self, "_active_observer", None)

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("router", self._router_node)
        builder.add_node("decline", self._decline_node)
        builder.add_node("profile_info", self._profile_info_node)
        builder.add_node("agent", self._agent_node)
        builder.add_node("profile_extractor", self._profile_extractor_node)
        builder.add_edge(START, "router")
        builder.add_conditional_edges(
            "router",
            _route_after_router,
            {
                "decline": "decline",
                "profile_info": "profile_info",
                "agent": "agent",
            },
        )
        builder.add_edge("decline", END)
        builder.add_edge("profile_info", END)
        builder.add_edge("agent", "profile_extractor")
        builder.add_edge("profile_extractor", END)
        return builder.compile(checkpointer=self._checkpointer)

    def _router_node(self, state: AgentState) -> dict[str, Any]:
        question = _latest_human_message(state["messages"])
        decision: RouteDecision = route_query(self.router_model, question)
        return {
            "query_type": decision.query_type,
            "route_reason": decision.reason,
            "token_usage": decision.token_usage,
        }

    def _decline_node(self, state: AgentState) -> dict[str, Any]:
        answer = (
            "I can only answer questions about the Bitext customer-service dataset: "
            "its categories, intents, examples, counts, distributions, and response patterns."
        )
        return {
            "messages": [AIMessage(content=answer)],
            "final_answer": answer,
            "proposed_profile_facts": [],
        }

    def _profile_info_node(self, state: AgentState) -> dict[str, Any]:
        facts = self.profile_repository.list_facts(self.settings.user_id)
        answer = format_profile_answer(facts)
        return {
            "messages": [AIMessage(content=answer)],
            "final_answer": answer,
            "proposed_profile_facts": [],
        }

    def _agent_node(
        self,
        state: AgentState,
        trace_callback: TraceCallback | None = None,
        observer: RunObserver | None = None,
    ) -> dict[str, Any]:
        trace_callback = self._current_trace_callback(trace_callback)
        observer = self._current_observer(observer)
        if hasattr(self, "toolbox"):
            self.toolbox.remember_filters(state.get("recent_filters", {}))
        result = self._run_agent_with_partial_trace(
            state,
            trace_callback=trace_callback,
            observer=observer,
        )
        result_messages = result["messages"]
        new_messages = result_messages[len(state["messages"]) :]
        final_answer = _last_ai_content(result_messages) or _max_iteration_message()
        return {
            "messages": new_messages,
            "final_answer": final_answer,
            "recent_filters": _recent_filters_from_messages(new_messages),
        }

    def _profile_extractor_node(self, state: AgentState) -> dict[str, Any]:
        if state.get("query_type") not in {"structured", "unstructured"}:
            return {"proposed_profile_facts": []}

        existing_facts = self.profile_repository.list_facts(self.settings.user_id)
        try:
            proposed_facts = extract_profile_facts(
                self.profile_model,
                existing_facts=existing_facts,
                user_messages=_all_user_messages(state.get("messages", [])),
                final_answer=state.get("final_answer", ""),
            )
        except Exception:
            proposed_facts = []
        return {"proposed_profile_facts": proposed_facts}

    def _run_agent_with_partial_trace(
        self,
        state: AgentState,
        trace_callback: TraceCallback | None = None,
        observer: RunObserver | None = None,
    ) -> dict[str, Any]:
        messages = list(state["messages"])
        seen_message_keys = {_message_trace_key(message) for message in messages}
        try:
            for chunk in self.agent.stream(
                {"messages": state["messages"]},
                config=self._graph_config(),
                stream_mode="values",
            ):
                for message in chunk.get("messages", []):
                    message_key = _message_trace_key(message)
                    if message_key not in seen_message_keys:
                        messages.append(message)
                        seen_message_keys.add(message_key)
                        _emit_trace_message(message, trace_callback, observer)
            return {"messages": messages}
        except GraphRecursionError:
            messages.append(AIMessage(content=_max_iteration_message()))
            return {"messages": messages}


def build_agent(settings: AgentSettings) -> BitextAgent:
    """Build the Bitext agent from settings."""

    dataset = BitextDataset.from_path_or_download(settings.dataset_path)
    return BitextAgent(settings, dataset)


def _make_chat_model(
    settings: AgentSettings,
    temperature: float,
    max_tokens: int,
    model_name: str | None = None,
) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=model_name or settings.model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=30,
    )


def _route_after_router(state: AgentState) -> str:
    if state.get("query_type") == "out_of_scope":
        return "decline"
    if state.get("query_type") == "profile_info":
        return "profile_info"
    return "agent"


def _merge_recent_filters(
    left: dict[str, dict[str, Any]] | None,
    right: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


def _emit_router_update(
    route_update: dict[str, Any],
    trace_callback: TraceCallback,
    observer: RunObserver | None,
    *,
    latency_ms: int,
) -> None:
    if observer is not None:
        observer.router_completed(
            route=route_update["query_type"],
            reason=route_update["route_reason"],
            latency_ms=latency_ms,
            token_usage=route_update.get("token_usage"),
        )

    trace_callback(
        {
            "type": "router",
            "query_type": route_update["query_type"],
            "reason": route_update["route_reason"],
        }
    )


def _recent_filters_from_messages(messages: list[AnyMessage]) -> dict[str, dict[str, Any]]:
    filters: dict[str, dict[str, Any]] = {}
    for message in messages:
        if not isinstance(message, ToolMessage) or getattr(message, "name", None) != "filter_dataset":
            continue
        payload = _json_message_content(message.content)
        filter_id = payload.get("filter_id")
        criteria = payload.get("criteria")
        if isinstance(filter_id, str) and isinstance(criteria, dict):
            filters[filter_id] = {
                "category": criteria.get("category"),
                "intent": criteria.get("intent"),
                "text_query": criteria.get("text_query"),
            }
    return filters


def _all_user_messages(messages: list[AnyMessage]) -> list[str]:
    return [str(message.content) for message in messages if isinstance(message, HumanMessage)]


def _json_message_content(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _latest_human_message(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    raise ValueError("No human message found in graph state.")


def _last_ai_content(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            return str(message.content)
    return ""


def _emit_trace_message(
    message: AnyMessage,
    trace_callback: TraceCallback | None,
    observer: RunObserver | None,
) -> None:
    if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
        for tool_call in message.tool_calls:
            tool_call_id = tool_call.get("id")
            args = tool_call.get("args", {})
            if trace_callback is not None:
                trace_callback(
                    {
                        "type": "tool_call",
                        "name": tool_call["name"],
                        "args": args,
                    }
                )
            if observer is not None:
                observer.tool_called(tool_call["name"], tool_call_id, args)
        if observer is not None:
            observer.model_completed(message)
    elif isinstance(message, ToolMessage):
        if trace_callback is not None:
            trace_callback(
                {
                    "type": "observation",
                    "name": getattr(message, "name"),
                    "content": message.content,
                }
            )
        if observer is not None:
            observer.tool_completed(
                getattr(message, "name", None),
                getattr(message, "tool_call_id", None),
                message.content,
            )
    elif observer is not None:
        observer.model_completed(message)


def _technical_problem_state(question: str, error: Exception) -> AgentState:
    return {
        "messages": [
            HumanMessage(content=question),
            AIMessage(content=TECHNICAL_PROBLEM_MESSAGE),
        ],
        "query_type": "error",
        "route_reason": str(error),
        "final_answer": TECHNICAL_PROBLEM_MESSAGE,
    }


def _attach_observability_state(
    state: AgentState,
    observer: RunObserver | None,
) -> AgentState:
    if observer is None:
        return state
    state["run_id"] = observer.run_id
    state["observability_path"] = str(observer.path)
    return state


def _drain_events(events: list[AgentEvent]) -> Iterator[AgentEvent]:
    while events:
        yield events.pop(0)


def _final_event(state: AgentState) -> AgentEvent:
    return {
        "type": "final",
        "state": state,
        "final_answer": state.get("final_answer", ""),
    }


def _elapsed_ms(started_at: float) -> int:
    return int((perf_counter() - started_at) * 1000)


def _router_error_trace_fields(error: RouterError) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    response_metadata = getattr(error, "response_metadata", None) or {}
    finish_reason = response_metadata.get("finish_reason")
    if finish_reason is not None:
        fields["finish_reason"] = finish_reason

    token_usage = getattr(error, "token_usage", None)
    if token_usage is not None:
        fields["token_usage"] = token_usage

    model_response = getattr(error, "model_response", None)
    if model_response is not None:
        fields["model_response_chars"] = len(model_response)
    return fields


def _message_trace_key(message: AnyMessage) -> tuple[Any, ...]:
    message_id = getattr(message, "id", None)
    if message_id:
        return (message.__class__.__name__, message_id)
    if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
        tool_call_ids = tuple(tool_call.get("id") for tool_call in message.tool_calls)
        return ("AIMessage", tool_call_ids, str(message.content))
    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        return (message.__class__.__name__, tool_call_id)
    return (message.__class__.__name__, str(message.content))


def _max_iteration_message() -> str:
    return (
        "I could not complete the dataset analysis within the configured iteration limit. "
        "Please narrow the question and try again."
    )
