"""Interactive CLI for the Bitext customer-service data analyst agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from uuid import uuid4

from bitext_agent.cli import UnknownSessionError, run_cli
from bitext_agent.config import AgentSettings


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(description="Run the Bitext data analyst agent.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Optional local CSV path. If omitted, the dataset is downloaded to data/.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=12,
        help="LangGraph recursion limit for one agent turn. Use 10-15 for the assignment.",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Optional conversation session id to resume from persistent memory.",
    )
    parser.add_argument(
        "--user",
        default="default",
        help="Optional user id for long-term profile memory. Defaults to 'default'.",
    )
    parser.add_argument(
        "--usage",
        action="store_true",
        help="Print command usage and exit without starting the agent.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.usage:
        parser.print_help()
        return 0

    session_id = args.session or f"session-{uuid4()}"
    settings = AgentSettings.from_env(
        dataset_path=args.dataset_path,
        max_iterations=args.max_iterations,
        session_id=session_id,
        resume_existing_session=args.session is not None,
        user_id=args.user,
    )
    try:
        run_cli(settings)
    except UnknownSessionError as error:
        print(error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
