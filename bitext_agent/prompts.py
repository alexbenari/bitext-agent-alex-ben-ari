"""Prompt loading for routing and answering Bitext dataset questions."""

from __future__ import annotations

from pathlib import Path


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(filename: str) -> str:
    """Load a prompt markdown file by name."""

    return (PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


ROUTER_PROMPT = load_prompt("router.md")
AGENT_SYSTEM_PROMPT = load_prompt("agent_system.md")
PROFILE_EXTRACTOR_PROMPT = load_prompt("profile_extractor.md")
