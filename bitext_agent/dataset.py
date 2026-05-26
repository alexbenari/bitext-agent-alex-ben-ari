"""Dataset loading and deterministic analysis helpers."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download


DATASET_REPO_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
DATASET_FILENAME = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
EXPECTED_COLUMNS = ("flags", "instruction", "category", "intent", "response")


def normalize_label(value: str | None) -> str | None:
    """Normalize a category or intent label for comparison."""

    if value is None:
        return None
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_")
    return normalized.upper() if normalized else None


def normalize_intent(value: str | None) -> str | None:
    """Normalize an intent label to the dataset's lower snake_case style."""

    if value is None:
        return None
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_")
    return normalized.lower() if normalized else None


@dataclass(frozen=True)
class DatasetFilter:
    """Reusable filter criteria for dataset tools."""

    category: str | None = None
    intent: str | None = None
    text_query: str | None = None

    def cache_key(self) -> str:
        payload = json.dumps(
            {
                "category": self.category,
                "intent": self.intent,
                "text_query": self.text_query,
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


class BitextDataset:
    """In-memory access layer for the Bitext customer-service CSV."""

    def __init__(self, frame: pd.DataFrame) -> None:
        missing_columns = [name for name in EXPECTED_COLUMNS if name not in frame.columns]
        if missing_columns:
            raise ValueError(f"Dataset is missing required columns: {missing_columns}")

        self._frame = frame.loc[:, EXPECTED_COLUMNS].copy()
        self._frame["category"] = self._frame["category"].astype(str).str.upper()
        self._frame["intent"] = self._frame["intent"].astype(str).str.lower()
        self._frame["instruction"] = self._frame["instruction"].astype(str)
        self._frame["response"] = self._frame["response"].astype(str)

    @classmethod
    def from_path_or_download(cls, dataset_path: Path) -> "BitextDataset":
        """Load a local CSV, downloading it from Hugging Face when needed."""

        if not dataset_path.exists():
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded_path = hf_hub_download(
                repo_id=DATASET_REPO_ID,
                filename=DATASET_FILENAME,
                repo_type="dataset",
            )
            shutil.copyfile(downloaded_path, dataset_path)

        return cls(pd.read_csv(dataset_path))

    @property
    def row_count(self) -> int:
        return int(len(self._frame))

    def schema_summary(self) -> dict[str, Any]:
        """Return dataset columns, row count, categories, and intents."""

        return {
            "row_count": self.row_count,
            "columns": list(self._frame.columns),
            "categories": self.categories(),
            "intents_by_category": self.intents_by_category(),
        }

    def categories(self) -> list[str]:
        """Return all dataset categories."""

        return sorted(self._frame["category"].dropna().unique().tolist())

    def intents_by_category(self) -> dict[str, list[str]]:
        """Return intents grouped by category."""

        grouped: dict[str, list[str]] = {}
        for category, group in self._frame.groupby("category", sort=True):
            grouped[str(category)] = sorted(group["intent"].dropna().unique().tolist())
        return grouped

    def validate_category(self, category: str | None) -> str | None:
        """Normalize and validate a category label against the dataset."""

        normalized = normalize_label(category)
        if normalized is None:
            return None
        categories = set(self.categories())
        if normalized in categories:
            return normalized

        raise ValueError(f"Unknown category '{category}'. Known categories: {sorted(categories)}")

    def resolve_intent(self, intent: str | None) -> str | None:
        """Resolve a user-supplied intent label to a known intent."""

        normalized = normalize_intent(intent)
        if normalized is None:
            return None
        intents = {intent for values in self.intents_by_category().values() for intent in values}
        if normalized in intents:
            return normalized
        raise ValueError(f"Unknown intent '{intent}'. Known intents: {sorted(intents)}")

    def filter_rows(self, criteria: DatasetFilter) -> pd.DataFrame:
        """Return rows matching category, intent, and free-text criteria."""

        filtered = self._frame
        category = self.validate_category(criteria.category)
        intent = self.resolve_intent(criteria.intent)

        if category is not None:
            filtered = filtered[filtered["category"] == category]
        if intent is not None:
            filtered = filtered[filtered["intent"] == intent]
        if criteria.text_query:
            literal_query = criteria.text_query.strip().lower()
            haystack = (
                filtered["instruction"].str.lower()
                + " "
                + filtered["response"].str.lower()
                + " "
                + filtered["intent"].str.lower()
                + " "
                + filtered["category"].str.lower()
            )
            filtered = filtered[haystack.str.contains(literal_query, regex=False)]

        return filtered

    def rows_by_ids(self, row_ids: list[int]) -> pd.DataFrame:
        """Return rows by positional row ids."""

        valid_ids = [row_id for row_id in row_ids if 0 <= row_id < len(self._frame)]
        return self._frame.iloc[valid_ids]

    def row_ids_for(self, frame: pd.DataFrame) -> list[int]:
        """Return positional row ids for a filtered frame."""

        return [int(index) for index in frame.index.tolist()]

    def examples(self, frame: pd.DataFrame, n: int) -> list[dict[str, Any]]:
        """Return compact example rows from a frame."""

        columns = ["category", "intent", "instruction", "response"]
        return frame.loc[:, columns].head(n).to_dict(orient="records")
