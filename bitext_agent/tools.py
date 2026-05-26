"""LangChain tools for deterministic Bitext dataset analysis."""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import wraps
from typing import Any, Optional

from pydantic import BaseModel, Field
from pydantic_core import ValidationError

from langchain_core.tools import StructuredTool, ToolException

from bitext_agent.dataset import BitextDataset, DatasetFilter


class DatasetSchemaInput(BaseModel):
    """Input for retrieving dataset schema metadata."""


class FilterDatasetInput(BaseModel):
    """Input for creating a reusable filtered dataset view."""

    category: Optional[str] = Field(
        default=None,
        description="Optional dataset category such as REFUND, ACCOUNT, FEEDBACK, or SHIPPING.",
    )
    intent: Optional[str] = Field(
        default=None,
        description="Optional dataset intent such as get_refund, complaint, or track_order.",
    )
    text_query: Optional[str] = Field(
        default=None,
        description=(
            "Optional literal words to search in instruction, response, intent, and category text. "
            "Do not use this for semantic matching; map natural language to category or intent first."
        ),
    )
    preview_rows: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of matching examples to include in the filter preview.",
    )


class CountRowsInput(BaseModel):
    """Input for counting rows directly or from a saved filter."""

    filter_id: Optional[str] = Field(
        default=None,
        description="Filter id returned by filter_dataset. Use this after filtering.",
    )
    category: Optional[str] = Field(default=None, description="Optional category to count directly.")
    intent: Optional[str] = Field(default=None, description="Optional intent to count directly.")
    text_query: Optional[str] = Field(default=None, description="Optional text query to count directly.")


class ShowExamplesInput(BaseModel):
    """Input for sampling dataset examples directly or from a saved filter."""

    filter_id: Optional[str] = Field(default=None, description="Filter id returned by filter_dataset.")
    category: Optional[str] = Field(default=None, description="Optional category to sample directly.")
    intent: Optional[str] = Field(default=None, description="Optional intent to sample directly.")
    text_query: Optional[str] = Field(default=None, description="Optional text query to sample directly.")
    n: int = Field(default=3, ge=1, le=10, description="Number of examples to return.")
    offset: int = Field(
        default=0,
        ge=0,
        description="Number of matching examples to skip before returning results.",
    )


class ExamplesByCategoryInput(BaseModel):
    """Input for sampling examples from every dataset category."""

    n_per_category: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Number of examples to return for each dataset category.",
    )


class IntentDistributionInput(BaseModel):
    """Input for computing intent counts over the dataset or a subset."""

    filter_id: Optional[str] = Field(default=None, description="Filter id returned by filter_dataset.")
    category: Optional[str] = Field(default=None, description="Optional category to group by intent.")
    text_query: Optional[str] = Field(default=None, description="Optional text query to filter before grouping.")


class ResponsePatternsInput(BaseModel):
    """Input for collecting examples used in qualitative response summaries."""

    category: Optional[str] = Field(default=None, description="Optional category to summarize.")
    intent: Optional[str] = Field(default=None, description="Optional intent to summarize.")
    text_query: Optional[str] = Field(default=None, description="Optional text query to summarize.")
    sample_size: int = Field(
        default=8,
        ge=3,
        le=20,
        description="Number of instruction/response pairs to collect for qualitative synthesis.",
    )


class DatasetToolbox:
    """Factory for dataset tools that share filter cache state."""

    def __init__(self, dataset: BitextDataset) -> None:
        self._dataset = dataset
        self._filter_cache: dict[str, list[int]] = {}
        self._filter_criteria: dict[str, DatasetFilter] = {}

    def get_dataset_schema(self) -> dict[str, Any]:
        """Return row count, columns, categories, and intents in the dataset."""

        return self._dataset.schema_summary()

    def filter_dataset(
        self,
        category: str | None = None,
        intent: str | None = None,
        text_query: str | None = None,
        preview_rows: int = 3,
    ) -> dict[str, Any]:
        """Create a reusable filtered dataset view and return its id, count, and examples."""

        criteria = DatasetFilter(category=category, intent=intent, text_query=text_query)
        frame = self._dataset.filter_rows(criteria)
        filter_id = criteria.cache_key()
        self._filter_cache[filter_id] = self._dataset.row_ids_for(frame)
        self._filter_criteria[filter_id] = criteria
        return {
            "filter_id": filter_id,
            "total_matches": int(len(frame)),
            "criteria": {
                "category": category,
                "intent": intent,
                "text_query": text_query,
            },
            "preview": self._dataset.examples(frame, preview_rows),
        }

    def count_rows(
        self,
        filter_id: str | None = None,
        category: str | None = None,
        intent: str | None = None,
        text_query: str | None = None,
    ) -> dict[str, Any]:
        """Count rows for a prior filter or direct filter criteria."""

        frame = self._frame_from_filter_or_criteria(filter_id, category, intent, text_query)
        return {"count": int(len(frame))}

    def show_examples(
        self,
        filter_id: str | None = None,
        category: str | None = None,
        intent: str | None = None,
        text_query: str | None = None,
        n: int = 3,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Return representative instruction/response examples."""

        frame = self._frame_from_filter_or_criteria(filter_id, category, intent, text_query)
        return {
            "count_available": int(len(frame)),
            "offset": offset,
            "examples": self._dataset.examples(frame.iloc[offset:], n),
        }

    def remember_filters(self, filters: dict[str, dict[str, Any]]) -> None:
        """Load persisted filter criteria into the in-process filter cache."""

        for filter_id, criteria_payload in filters.items():
            criteria = DatasetFilter(
                category=criteria_payload.get("category"),
                intent=criteria_payload.get("intent"),
                text_query=criteria_payload.get("text_query"),
            )
            self._filter_criteria[filter_id] = criteria

    def examples_by_category(self, n_per_category: int = 1) -> dict[str, Any]:
        """Return examples grouped by every dataset category."""

        examples_by_category: dict[str, list[dict[str, Any]]] = {}
        for category in self._dataset.categories():
            frame = self._dataset.filter_rows(DatasetFilter(category=category))
            examples_by_category[category] = self._dataset.examples(frame, n_per_category)

        return {
            "category_count": len(examples_by_category),
            "examples_per_category": n_per_category,
            "examples_by_category": examples_by_category,
        }

    def intent_distribution(
        self,
        filter_id: str | None = None,
        category: str | None = None,
        text_query: str | None = None,
    ) -> dict[str, Any]:
        """Return counts by intent for a category, text query, or prior filter."""

        frame = self._frame_from_filter_or_criteria(filter_id, category, None, text_query)
        counts = frame["intent"].value_counts().sort_index()
        return {
            "total_rows": int(len(frame)),
            "intent_counts": {str(intent): int(count) for intent, count in counts.items()},
        }

    def collect_response_patterns(
        self,
        category: str | None = None,
        intent: str | None = None,
        text_query: str | None = None,
        sample_size: int = 8,
    ) -> dict[str, Any]:
        """Collect rows and aggregates for qualitative response summaries."""

        frame = self._frame_from_filter_or_criteria(None, category, intent, text_query)
        intent_counts = frame["intent"].value_counts().head(10)
        return {
            "total_rows": int(len(frame)),
            "top_intents": {str(intent): int(count) for intent, count in intent_counts.items()},
            "samples": self._dataset.examples(frame, sample_size),
        }

    def tools(self) -> list[Any]:
        """Build LangChain StructuredTool objects with Pydantic schemas."""

        return [
            _structured_dataset_tool(
                name="get_dataset_schema",
                description=(
                    "Use to list dataset columns, total row count, all categories, and all intents. "
                    "Best for questions like 'what categories exist?' or 'what intents are in REFUND?'."
                ),
                func=self.get_dataset_schema,
                args_schema=DatasetSchemaInput,
            ),
            _structured_dataset_tool(
                name="filter_dataset",
                description=(
                    "Use first when a question restricts the dataset by category, intent, or literal text. "
                    "Returns a filter_id that later tools can use for counts, examples, or distributions."
                ),
                func=self.filter_dataset,
                args_schema=FilterDatasetInput,
            ),
            _structured_dataset_tool(
                name="count_rows",
                description=(
                    "Use to count dataset rows. Prefer passing filter_id after filter_dataset for "
                    "multi-step questions like counting refund requests."
                ),
                func=self.count_rows,
                args_schema=CountRowsInput,
            ),
            _structured_dataset_tool(
                name="show_examples",
                description=(
                    "Use to show example customer instructions and support responses from a prior "
                    "filter_id or direct category/intent/text criteria."
                ),
                func=self.show_examples,
                args_schema=ShowExamplesInput,
            ),
            _structured_dataset_tool(
                name="examples_by_category",
                description=(
                    "Use when the user asks for one or more examples from each category, "
                    "all categories, or every category. Returns examples grouped by category "
                    "in one tool call."
                ),
                func=self.examples_by_category,
                args_schema=ExamplesByCategoryInput,
            ),
            _structured_dataset_tool(
                name="intent_distribution",
                description=(
                    "Use to compute the distribution of intents overall, inside a category, or within "
                    "a previously returned filter_id. Do not estimate distributions manually."
                ),
                func=self.intent_distribution,
                args_schema=IntentDistributionInput,
            ),
            _structured_dataset_tool(
                name="collect_response_patterns",
                description=(
                    "Use for open-ended dataset-grounded summaries about how customers ask or how "
                    "agents respond. Returns counts and sample instruction/response pairs."
                ),
                func=self.collect_response_patterns,
                args_schema=ResponsePatternsInput,
            ),
        ]

    def _frame_from_filter_or_criteria(
        self,
        filter_id: str | None,
        category: str | None,
        intent: str | None,
        text_query: str | None,
    ):
        if filter_id is not None:
            if filter_id not in self._filter_cache:
                self._rebuild_filter_cache(filter_id)
            return self._dataset.rows_by_ids(self._filter_cache[filter_id])

        criteria = DatasetFilter(category=category, intent=intent, text_query=text_query)
        return self._dataset.filter_rows(criteria)

    def _rebuild_filter_cache(self, filter_id: str) -> None:
        if filter_id not in self._filter_criteria:
            raise ValueError(
                f"Unknown filter_id '{filter_id}'. Call filter_dataset before using it."
            )

        frame = self._dataset.filter_rows(self._filter_criteria[filter_id])
        self._filter_cache[filter_id] = self._dataset.row_ids_for(frame)


def _structured_dataset_tool(
    *,
    name: str,
    description: str,
    func: Callable[..., Any],
    args_schema: type[BaseModel],
) -> StructuredTool:
    return StructuredTool.from_function(
        name=name,
        description=description,
        func=_raise_tool_exception_for_recoverable_errors(func),
        args_schema=args_schema,
        handle_tool_error=_format_tool_exception,
        handle_validation_error=_format_validation_error,
    )


def _raise_tool_exception_for_recoverable_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except ValueError as error:
            raise ToolException(str(error)) from error

    return wrapped


def _format_tool_exception(error: ToolException) -> str:
    return _json_error_observation("recoverable_tool_error", str(error))


def _format_validation_error(error: ValidationError) -> str:
    return _json_error_observation("tool_validation_error", str(error))


def _json_error_observation(error_type: str, message: str) -> str:
    return json.dumps(
        {
            "ok": False,
            "error": {
                "type": error_type,
                "message": message,
            },
        },
        ensure_ascii=False,
    )
