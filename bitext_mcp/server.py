"""FastMCP server exposing deterministic Bitext dataset tools."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from bitext_agent.config import DEFAULT_DATASET_CACHE
from bitext_agent.dataset import BitextDataset
from bitext_agent.tools import (
    CountRowsInput,
    DatasetToolbox,
    ExamplesByCategoryInput,
    FilterDatasetInput,
    IntentDistributionInput,
    ResponsePatternsInput,
    ShowExamplesInput,
)


def create_server(dataset: BitextDataset | None = None) -> FastMCP:
    """Create a FastMCP server for Bitext dataset analysis tools."""

    loaded_dataset = dataset or BitextDataset.from_path_or_download(DEFAULT_DATASET_CACHE)
    toolbox = DatasetToolbox(loaded_dataset)
    mcp = FastMCP("Bitext Dataset Tools")

    @mcp.tool
    def get_dataset_schema() -> dict[str, Any]:
        """Return row count, columns, categories, and intents in the dataset."""

        return toolbox.get_dataset_schema()

    @mcp.tool
    def filter_dataset(
        category: str | None = None,
        intent: str | None = None,
        text_query: str | None = None,
        preview_rows: int = 3,
    ) -> dict[str, Any]:
        """Create a reusable filtered dataset view and return its id, count, and examples."""

        args = FilterDatasetInput(
            category=category,
            intent=intent,
            text_query=text_query,
            preview_rows=preview_rows,
        )
        return toolbox.filter_dataset(**args.model_dump())

    @mcp.tool
    def count_rows(
        filter_id: str | None = None,
        category: str | None = None,
        intent: str | None = None,
        text_query: str | None = None,
    ) -> dict[str, Any]:
        """Count rows for a prior filter or direct filter criteria."""

        args = CountRowsInput(
            filter_id=filter_id,
            category=category,
            intent=intent,
            text_query=text_query,
        )
        return toolbox.count_rows(**args.model_dump())

    @mcp.tool
    def show_examples(
        filter_id: str | None = None,
        category: str | None = None,
        intent: str | None = None,
        text_query: str | None = None,
        n: int = 3,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Return representative instruction and response examples."""

        args = ShowExamplesInput(
            filter_id=filter_id,
            category=category,
            intent=intent,
            text_query=text_query,
            n=n,
            offset=offset,
        )
        return toolbox.show_examples(**args.model_dump())

    @mcp.tool
    def examples_by_category(n_per_category: int = 1) -> dict[str, Any]:
        """Return examples grouped by every dataset category."""

        args = ExamplesByCategoryInput(n_per_category=n_per_category)
        return toolbox.examples_by_category(**args.model_dump())

    @mcp.tool
    def intent_distribution(
        filter_id: str | None = None,
        category: str | None = None,
        text_query: str | None = None,
    ) -> dict[str, Any]:
        """Return counts by intent for all rows, a category, text query, or prior filter."""

        args = IntentDistributionInput(
            filter_id=filter_id,
            category=category,
            text_query=text_query,
        )
        return toolbox.intent_distribution(**args.model_dump())

    @mcp.tool
    def collect_response_patterns(
        category: str | None = None,
        intent: str | None = None,
        text_query: str | None = None,
        sample_size: int = 8,
    ) -> dict[str, Any]:
        """Collect rows and aggregates for qualitative response summaries."""

        args = ResponsePatternsInput(
            category=category,
            intent=intent,
            text_query=text_query,
            sample_size=sample_size,
        )
        return toolbox.collect_response_patterns(**args.model_dump())

    return mcp


def main() -> None:
    """Run the Bitext MCP server with FastMCP's default transport."""

    create_server().run()


if __name__ == "__main__":
    main()
