from __future__ import annotations

import asyncio
import warnings

import pandas as pd
import pytest
from authlib.deprecate import AuthlibDeprecationWarning


warnings.filterwarnings(
    "ignore",
    category=AuthlibDeprecationWarning,
    module="fastmcp.server.auth.providers.jwt",
)

from fastmcp import Client
from fastmcp.exceptions import ToolError

from bitext_agent.dataset import BitextDataset
from bitext_mcp.server import create_server
from examples.mcp_client_example import inspect_server_and_list_categories


EXPECTED_TOOL_NAMES = {
    "get_dataset_schema",
    "filter_dataset",
    "count_rows",
    "show_examples",
    "examples_by_category",
    "intent_distribution",
    "collect_response_patterns",
}


def make_dataset() -> BitextDataset:
    return BitextDataset(
        pd.DataFrame(
            [
                {
                    "flags": "B",
                    "instruction": "I want my money back",
                    "category": "REFUND",
                    "intent": "get_refund",
                    "response": "I can help you request a refund.",
                },
                {
                    "flags": "B",
                    "instruction": "I need a refund update",
                    "category": "REFUND",
                    "intent": "track_refund",
                    "response": "I can help you track your refund.",
                },
                {
                    "flags": "B",
                    "instruction": "Where is my order?",
                    "category": "ORDER",
                    "intent": "track_order",
                    "response": "You can track your order from your account.",
                },
                {
                    "flags": "P",
                    "instruction": "I have a complaint",
                    "category": "FEEDBACK",
                    "intent": "complaint",
                    "response": "I am sorry to hear that and can escalate this.",
                },
            ]
        )
    )


def test_mcp_server_registers_dataset_tools() -> None:
    async def run() -> set[str]:
        server = create_server(make_dataset())
        async with Client(server) as client:
            tools = await client.list_tools()
        return {tool.name for tool in tools}

    assert run_async(run()) == EXPECTED_TOOL_NAMES


def test_mcp_tool_schemas_keep_public_adapter_contract_stable() -> None:
    async def run() -> dict[str, set[str]]:
        server = create_server(make_dataset())
        async with Client(server) as client:
            tools = await client.list_tools()
        return {
            tool.name: set(tool.inputSchema.get("properties", {}))
            for tool in tools
            if tool.name in EXPECTED_TOOL_NAMES
        }

    assert run_async(run()) == {
        "get_dataset_schema": set(),
        "filter_dataset": {"category", "intent", "text_query", "preview_rows"},
        "count_rows": {"filter_id", "category", "intent", "text_query"},
        "show_examples": {"filter_id", "category", "intent", "text_query", "n", "offset"},
        "examples_by_category": {"n_per_category"},
        "intent_distribution": {"filter_id", "category", "text_query"},
        "collect_response_patterns": {"category", "intent", "text_query", "sample_size"},
    }


def test_mcp_client_can_call_count_rows_by_public_arguments() -> None:
    async def run() -> dict[str, int]:
        server = create_server(make_dataset())
        async with Client(server) as client:
            result = await client.call_tool("count_rows", {"category": "REFUND"})
        return result.data

    assert run_async(run()) == {"count": 2}


def test_mcp_filter_id_can_be_reused_across_adapter_tool_calls() -> None:
    async def run() -> dict[str, int]:
        server = create_server(make_dataset())
        async with Client(server) as client:
            filtered = await client.call_tool("filter_dataset", {"category": "REFUND"})
            counted = await client.call_tool(
                "count_rows",
                {"filter_id": filtered.data["filter_id"]},
            )
        return counted.data

    assert run_async(run()) == {"count": 2}


def test_mcp_show_examples_rejects_negative_n() -> None:
    async def run() -> None:
        server = create_server(make_dataset())
        async with Client(server) as client:
            await client.call_tool("show_examples", {"n": -1})

    with pytest.raises(ToolError, match="greater than or equal to 1"):
        run_async(run())


def test_mcp_show_examples_rejects_negative_offset() -> None:
    async def run() -> None:
        server = create_server(make_dataset())
        async with Client(server) as client:
            await client.call_tool("show_examples", {"offset": -1})

    with pytest.raises(ToolError, match="greater than or equal to 0"):
        run_async(run())


def test_create_server_does_not_require_nebius_key(monkeypatch) -> None:
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)

    server = create_server(make_dataset())

    assert server is not None


def test_sample_mcp_client_launches_server_lists_tools_and_categories() -> None:
    tool_names, schema = run_async(inspect_server_and_list_categories())

    assert set(tool_names) == EXPECTED_TOOL_NAMES
    assert "categories" in schema
    assert {"REFUND", "ORDER", "SHIPPING"}.issubset(set(schema["categories"]))


def run_async(coro):
    return asyncio.run(coro)
