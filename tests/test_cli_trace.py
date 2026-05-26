from __future__ import annotations

from langchain_core.messages import ToolMessage

from bitext_agent.cli import (
    UnknownSessionError,
    _configure_utf8_output,
    _print_live_trace_event,
    _print_trace,
    _styled,
    run_cli,
)
from bitext_agent.config import AgentSettings


class RecordingStream:
    def __init__(self) -> None:
        self.reconfigure_calls: list[dict[str, str]] = []

    def reconfigure(self, **kwargs) -> None:
        self.reconfigure_calls.append(kwargs)


class ExistingSessionAgent:
    def __init__(self) -> None:
        self.saved_facts: list[dict] = []
        self.profile_repository = self

    def has_saved_session(self) -> bool:
        return True

    def ask_live(self, *_args, **_kwargs):
        return {"final_answer": "done"}

    def stream(self, *_args, **_kwargs):
        yield {"type": "final", "state": self.ask_live()}

    def add_facts(self, user_id, facts, *, source_session_id):
        self.saved_facts.append(
            {
                "user_id": user_id,
                "facts": facts,
                "source_session_id": source_session_id,
            }
        )
        return len(facts)


class MissingSessionAgent:
    def has_saved_session(self) -> bool:
        return False


def test_run_cli_prints_generated_session_message(monkeypatch, capsys, tmp_path) -> None:
    settings = AgentSettings(
        api_key="test-key",
        base_url="https://example.com/v1/",
        model_name="test-model",
        dataset_path=tmp_path / "dataset.csv",
        max_iterations=12,
        session_id="session-generated",
        checkpoint_path=tmp_path / "checkpoints.sqlite",
    )
    monkeypatch.setattr("bitext_agent.cli.build_agent", lambda _settings: ExistingSessionAgent())
    monkeypatch.setattr("builtins.input", lambda _prompt: "exit")

    run_cli(settings)

    captured = capsys.readouterr().out
    assert "This conversation's session id is: session-generated." in captured
    assert "You can use it to resume it anytime after it ends." in captured
    assert "User profile: default" in captured


def test_run_cli_prints_resumed_session_message(monkeypatch, capsys, tmp_path) -> None:
    settings = AgentSettings(
        api_key="test-key",
        base_url="https://example.com/v1/",
        model_name="test-model",
        dataset_path=tmp_path / "dataset.csv",
        max_iterations=12,
        session_id="saved-session",
        resume_existing_session=True,
        checkpoint_path=tmp_path / "checkpoints.sqlite",
    )
    monkeypatch.setattr("bitext_agent.cli.build_agent", lambda _settings: ExistingSessionAgent())
    monkeypatch.setattr("builtins.input", lambda _prompt: "exit")

    run_cli(settings)

    captured = capsys.readouterr().out
    assert "This conversation is resumed from session id: saved-session" in captured


def test_run_cli_rejects_missing_explicit_session(monkeypatch, tmp_path) -> None:
    settings = AgentSettings(
        api_key="test-key",
        base_url="https://example.com/v1/",
        model_name="test-model",
        dataset_path=tmp_path / "dataset.csv",
        max_iterations=12,
        session_id="missing-session",
        resume_existing_session=True,
        checkpoint_path=tmp_path / "checkpoints.sqlite",
    )
    monkeypatch.setattr("bitext_agent.cli.build_agent", lambda _settings: MissingSessionAgent())

    try:
        run_cli(settings)
    except UnknownSessionError as error:
        assert "No saved conversation was found for session id: missing-session" in str(error)
        assert "start a new conversation without --session" in str(error)
    else:
        raise AssertionError("Expected missing explicit session to be rejected.")


def test_run_cli_approves_profile_facts(monkeypatch, capsys, tmp_path) -> None:
    settings = AgentSettings(
        api_key="test-key",
        base_url="https://example.com/v1/",
        model_name="test-model",
        dataset_path=tmp_path / "dataset.csv",
        max_iterations=12,
        session_id="session-1",
        user_id="alex",
        checkpoint_path=tmp_path / "checkpoints.sqlite",
    )
    agent = ExistingSessionAgent()
    agent.ask_live = lambda *_args, **_kwargs: {
        "final_answer": "done",
        "proposed_profile_facts": ["You prefer refund analytics."],
    }
    inputs = iter(["Show refunds", "y", "exit"])
    monkeypatch.setattr("bitext_agent.cli.build_agent", lambda _settings: agent)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    run_cli(settings)

    captured = capsys.readouterr().out
    assert "I can add this to your profile:" in captured
    assert "Saved 1 profile fact(s)." in captured
    assert agent.saved_facts == [
        {
            "user_id": "alex",
            "facts": ["You prefer refund analytics."],
            "source_session_id": "session-1",
        }
    ]


def test_run_cli_can_edit_profile_facts_before_saving(monkeypatch, tmp_path) -> None:
    settings = AgentSettings(
        api_key="test-key",
        base_url="https://example.com/v1/",
        model_name="test-model",
        dataset_path=tmp_path / "dataset.csv",
        max_iterations=12,
        session_id="session-1",
        user_id="alex",
        checkpoint_path=tmp_path / "checkpoints.sqlite",
    )
    agent = ExistingSessionAgent()
    agent.ask_live = lambda *_args, **_kwargs: {
        "final_answer": "done",
        "proposed_profile_facts": ["You prefer refund analytics."],
    }
    inputs = iter(["Show refunds", "edit", "You prefer examples.", "", "exit"])
    monkeypatch.setattr("bitext_agent.cli.build_agent", lambda _settings: agent)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    run_cli(settings)

    assert agent.saved_facts[0]["facts"] == ["You prefer examples."]


def test_configure_utf8_output_sets_stdout_and_stderr(monkeypatch) -> None:
    stdout = RecordingStream()
    stderr = RecordingStream()
    monkeypatch.setattr("bitext_agent.cli.sys.stdout", stdout)
    monkeypatch.setattr("bitext_agent.cli.sys.stderr", stderr)

    _configure_utf8_output()

    assert stdout.reconfigure_calls == [{"encoding": "utf-8", "errors": "replace"}]
    assert stderr.reconfigure_calls == [{"encoding": "utf-8", "errors": "replace"}]


def test_print_trace_outputs_full_tool_observation(capsys) -> None:
    long_observation = "prefix " + ("0123456789" * 90) + " suffix"
    state = {
        "query_type": "structured",
        "route_reason": "Dataset question.",
        "messages": [
            ToolMessage(
                content=long_observation,
                name="show_examples",
                tool_call_id="tool-call-1",
            )
        ],
    }

    _print_trace(state)

    captured = capsys.readouterr().out
    assert long_observation in captured
    assert "..." not in captured


def test_print_live_trace_event_formats_trace_sections(capsys) -> None:
    _print_live_trace_event(
        {
            "type": "tool_call",
            "name": "show_examples",
            "args": {"category": "REFUND", "n": 1},
        },
        use_color=False,
    )

    captured = capsys.readouterr().out
    assert "[trace/tool-call]" in captured
    assert "name: show_examples" in captured
    assert 'args: {"category": "REFUND", "n": 1}' in captured


def test_print_live_trace_event_greys_trace_fields_when_color_is_enabled(capsys) -> None:
    _print_live_trace_event(
        {
            "type": "tool_call",
            "name": "show_examples",
            "args": {"category": "REFUND", "n": 1},
        },
        use_color=True,
    )

    captured = capsys.readouterr().out
    assert "\033[90m[trace/tool-call]\033[0m" in captured
    assert "\033[90mname: show_examples\033[0m" in captured
    assert '\033[90margs: {"category": "REFUND", "n": 1}\033[0m' in captured


def test_print_live_trace_event_formats_error_section(capsys) -> None:
    _print_live_trace_event(
        {
            "type": "error",
            "stage": "router",
            "message": "router response invalid",
        },
        use_color=False,
    )

    captured = capsys.readouterr().out
    assert "[trace/error]" in captured
    assert "stage: router" in captured
    assert "message: router response invalid" in captured


def test_print_live_trace_event_formats_error_diagnostics(capsys) -> None:
    _print_live_trace_event(
        {
            "type": "error",
            "stage": "router",
            "message": "Router response did not contain JSON.",
            "finish_reason": "length",
            "token_usage": {"output_tokens": 256},
            "model_response_chars": 0,
        },
        use_color=False,
    )

    captured = capsys.readouterr().out
    assert "finish_reason: length" in captured
    assert 'token_usage: {"output_tokens": 256}' in captured
    assert "model_response_chars: 0" in captured


def test_styled_can_emit_ansi_formatting() -> None:
    assert _styled("[trace/router]", "\033[2;36m", use_color=True) == (
        "\033[2;36m[trace/router]\033[0m"
    )
