from __future__ import annotations

import main as cli_main


def test_usage_flag_prints_help_without_starting_agent(monkeypatch, capsys) -> None:
    started = False

    def fake_run_cli(_settings):
        nonlocal started
        started = True

    monkeypatch.setattr(cli_main, "run_cli", fake_run_cli)

    exit_code = cli_main.main(["--usage"])

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "usage:" in captured
    assert "--session" in captured
    assert "--user" in captured
    assert started is False


def test_parse_args_defaults_usage_to_false() -> None:
    args = cli_main.parse_args([])

    assert args.usage is False
