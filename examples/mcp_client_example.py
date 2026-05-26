"""Run a small FastMCP client against the Bitext MCP server."""

from __future__ import annotations

import asyncio
import json
import os
import warnings
from pathlib import Path
from typing import Any

from authlib.deprecate import AuthlibDeprecationWarning

warnings.filterwarnings(
    "ignore",
    category=AuthlibDeprecationWarning,
    module="fastmcp.server.auth.providers.jwt",
)

from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport


SERVER_SCRIPT = Path(__file__).resolve().parents[1] / "bitext_mcp" / "server.py"


def build_server_transport(log_path: Path | None = None) -> PythonStdioTransport:
    """Build a quiet stdio transport that launches the local MCP server."""

    server_env = {
        **os.environ,
        "FASTMCP_LOG_LEVEL": "ERROR",
        "FASTMCP_SHOW_BANNER": "false",
    }
    return PythonStdioTransport(
        SERVER_SCRIPT,
        env=server_env,
        keep_alive=False,
        log_file=log_path or Path(os.devnull),
    )


async def inspect_server_and_list_categories() -> tuple[list[str], dict[str, Any]]:
    """Launch the MCP server, list tools, and call get_dataset_schema."""

    async with Client(build_server_transport()) as client:
        tools = await client.list_tools()
        result = await client.call_tool("get_dataset_schema", {})
    return sorted(tool.name for tool in tools), result.data


def main() -> int:
    """Run the sample MCP client from the command line."""

    print("Starting Bitext MCP server...")
    tool_names, schema = asyncio.run(inspect_server_and_list_categories())
    print("MCP server started successfully.")
    print()
    print("Calling MCP client list_tools() to discover available tools...")
    print("Available MCP tools:")
    for tool_name in tool_names:
        print(f"- {tool_name}")
    print()
    print("Listing all dataset categories...")
    print("Calling MCP tool get_dataset_schema:")
    print(json.dumps(schema["categories"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
