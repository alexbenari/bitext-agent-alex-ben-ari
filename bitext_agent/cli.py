"""Command-line interface helpers for the Bitext agent."""

from __future__ import annotations

import json
import sys
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from bitext_agent.config import AgentSettings
from bitext_agent.graph import build_agent
from bitext_agent.observability import JsonlRunObserver


TRACE_STYLE = "\033[90m"
RESET_STYLE = "\033[0m"


class UnknownSessionError(RuntimeError):
    """Raised when a requested persistent session has no checkpoints."""


def run_cli(settings: AgentSettings) -> None:
    """Run an interactive conversation loop."""

    _configure_utf8_output()
    agent = build_agent(settings)
    try:
        if settings.resume_existing_session and not agent.has_saved_session():
            raise UnknownSessionError(
                f"No saved conversation was found for session id: {settings.session_id}\n"
                "Please verify the session id you passed, or start a new conversation without --session."
            )

        print(
            "Hi. I am your Bitext dataset analyst. Ask me anything about the dataset. Type 'exit' or 'quit' to stop."
        )
        if settings.resume_existing_session:
            print(f"This conversation is resumed from session id: {settings.session_id}")
        else:
            print(
                f"This conversation's session id is: {settings.session_id}. "
                "You can use it to resume it anytime after it ends."
            )
        print(f"User profile: {settings.user_id}")
        while True:
            question = input("\nYou: ").strip()
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue

            observer = JsonlRunObserver()
            state: dict[str, Any] = {}
            for event in agent.stream(question, observer=observer):
                if event.get("type") == "final":
                    state = event["state"]
                else:
                    _print_live_trace_event(event)
            print("\nAgent:")
            print(state.get("final_answer", ""))
            _handle_profile_fact_approval(agent, settings, state)
    finally:
        close = getattr(agent, "close", None)
        if close is not None:
            close()


def _configure_utf8_output() -> None:
    """Make CLI printing robust when model/tool output contains Unicode."""

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _handle_profile_fact_approval(agent, settings: AgentSettings, state: dict[str, Any]) -> None:
    proposed_facts = state.get("proposed_profile_facts") or []
    if not proposed_facts:
        return

    print("\nI can add this to your profile:")
    for index, fact in enumerate(proposed_facts, start=1):
        print(f"{index}. {fact}")

    choice = input("\nAdd these to your profile? [y/n/edit]: ").strip().lower()
    if choice == "y":
        facts_to_save = proposed_facts
    elif choice == "edit":
        facts_to_save = _read_edited_profile_facts()
    else:
        return

    inserted = agent.profile_repository.add_facts(
        settings.user_id,
        facts_to_save,
        source_session_id=settings.session_id,
    )
    if inserted:
        print(f"Saved {inserted} profile fact(s).")


def _read_edited_profile_facts() -> list[str]:
    print("Enter replacement profile facts, one per line. Submit a blank line when done.")
    facts: list[str] = []
    while True:
        fact = input("Profile fact: ").strip()
        if not fact:
            break
        facts.append(fact)
    return facts


def _print_trace(state: dict[str, Any]) -> None:
    print()
    _print_live_trace_event(
        {
            "type": "router",
            "query_type": state.get("query_type"),
            "reason": state.get("route_reason"),
        },
        use_color=False,
    )

    for message in state.get("messages", []):
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            for tool_call in message.tool_calls:
                _print_live_trace_event(
                    {
                        "type": "tool_call",
                        "name": tool_call["name"],
                        "args": tool_call.get("args", {}),
                    },
                    use_color=False,
                )
        elif isinstance(message, ToolMessage):
            _print_live_trace_event(
                {
                    "type": "observation",
                    "name": message.name,
                    "content": message.content,
                },
                use_color=False,
            )


def _print_live_trace_event(event: dict[str, Any], use_color: bool | None = None) -> None:
    if use_color is None:
        use_color = sys.stdout.isatty()

    event_type = event.get("type")
    if event_type == "router":
        print(f"\n{_styled('[trace/router]', TRACE_STYLE, use_color)}", flush=True)
        _print_trace_field("type", event.get("query_type"), use_color)
        _print_trace_field("reason", event.get("reason"), use_color)
    elif event_type == "tool_call":
        print(f"\n{_styled('[trace/tool-call]', TRACE_STYLE, use_color)}", flush=True)
        _print_trace_field("name", event.get("name"), use_color)
        _print_trace_field("args", _compact_json(event.get("args", {})), use_color)
    elif event_type == "observation":
        print(f"\n{_styled('[trace/observation]', TRACE_STYLE, use_color)}", flush=True)
        _print_trace_field("tool", event.get("name"), use_color)
        _print_trace_field("output", _format_observation(event.get("content")), use_color)
    elif event_type == "error":
        print(f"\n{_styled('[trace/error]', TRACE_STYLE, use_color)}", flush=True)
        _print_trace_field("stage", event.get("stage"), use_color)
        _print_trace_field("message", event.get("message"), use_color)
        if "finish_reason" in event:
            _print_trace_field("finish_reason", event.get("finish_reason"), use_color)
        if "token_usage" in event:
            _print_trace_field("token_usage", _compact_json(event.get("token_usage")), use_color)
        if "model_response_chars" in event:
            _print_trace_field(
                "model_response_chars",
                event.get("model_response_chars"),
                use_color,
            )


def _print_trace_field(name: str, value: Any, use_color: bool) -> None:
    print(_styled(f"{name}: {value}", TRACE_STYLE, use_color), flush=True)


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _format_observation(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return " ".join(text.split())


def _styled(text: str, style: str, use_color: bool | None = None) -> str:
    if use_color is None:
        use_color = sys.stdout.isatty()
    return f"{style}{text}{RESET_STYLE}" if use_color else text
