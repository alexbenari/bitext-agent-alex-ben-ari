"""Configuration for the Bitext data analyst agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_MODEL = "nvidia/Nemotron-3-Nano-Omni"
DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_DATASET_CACHE = Path("data/bitext_customer_support.csv")
DEFAULT_CHECKPOINT_DB = Path("data/checkpoints.sqlite")
DEFAULT_PROFILE_DB = Path("data/user_profiles.sqlite")


@dataclass(frozen=True)
class AgentSettings:
    """Runtime settings required to build and run the agent."""

    api_key: str
    base_url: str
    model_name: str
    dataset_path: Path
    max_iterations: int
    session_id: str = "default"
    resume_existing_session: bool = False
    checkpoint_path: Path = DEFAULT_CHECKPOINT_DB
    user_id: str = "default"
    profile_path: Path = DEFAULT_PROFILE_DB

    @classmethod
    def from_env(
        cls,
        dataset_path: Path | None = None,
        max_iterations: int = 12,
        session_id: str = "default",
        resume_existing_session: bool = False,
        user_id: str = "default",
    ) -> "AgentSettings":
        """Create settings from environment variables and CLI overrides."""

        load_dotenv()
        api_key = os.getenv("NEBIUS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "NEBIUS_API_KEY is required. Set it to your Nebius Token Factory API key."
            )

        if not 10 <= max_iterations <= 15:
            raise ValueError("--max-iterations must be between 10 and 15 for Task 1.")

        return cls(
            api_key=api_key,
            base_url=os.getenv("NEBIUS_BASE_URL", DEFAULT_BASE_URL),
            model_name=os.getenv("NEBIUS_MODEL", DEFAULT_MODEL),
            dataset_path=dataset_path or DEFAULT_DATASET_CACHE,
            max_iterations=max_iterations,
            session_id=session_id,
            resume_existing_session=resume_existing_session,
            checkpoint_path=Path(os.getenv("BITEXT_CHECKPOINT_DB", DEFAULT_CHECKPOINT_DB)),
            user_id=user_id,
            profile_path=Path(os.getenv("BITEXT_PROFILE_DB", DEFAULT_PROFILE_DB)),
        )
