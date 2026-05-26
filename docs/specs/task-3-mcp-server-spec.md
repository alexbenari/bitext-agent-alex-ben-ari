# Task 3 MCP Server Spec

## Goal

Add a FastMCP server that exposes the existing Bitext dataset analysis tools over MCP. The server should let an MCP client inspect schema information, create filters, count rows, show examples, group examples by category, compute intent distributions, and collect response-pattern samples.

This feature satisfies Task 3 of the assignment: build an MCP server with FastMCP, expose at least three tools, and document how to start the server and call a tool from a client.

## Scope

In scope:

- Add a new `bitext_mcp` Python package.
- Implement a FastMCP server entry point in `bitext_mcp/server.py`.
- Reuse `bitext_agent.dataset.BitextDataset` for dataset loading.
- Reuse `bitext_agent.tools.DatasetToolbox` for all dataset-analysis behavior.
- Expose these MCP tools:
  - `get_dataset_schema`
  - `filter_dataset`
  - `count_rows`
  - `show_examples`
  - `examples_by_category`
  - `intent_distribution`
  - `collect_response_patterns`
- Add `fastmcp` as a project dependency.
- Update `README.md` with:
  - MCP server startup command.
  - A short client example that connects and calls one tool.
  - A short note that the MCP server exposes dataset tools only.
- Add tests for server construction and tool behavior.

Out of scope:

- Exposing the full LangGraph agent as an MCP tool.
- Adding Streamlit UI or query recommendation bonus features.
- Moving `dataset.py` or `tools.py` into a separate shared package.
- Changing the CLI behavior.
- Reworking memory, profile, routing, or prompt logic.

## Architecture

The MCP server will be a thin adapter over the existing deterministic tool layer:

```text
MCP client
  -> FastMCP tool function in bitext_mcp.server
    -> DatasetToolbox method
      -> BitextDataset
        -> local CSV or Hugging Face download
```

The adapter functions will use FastMCP's tool registration style and delegate actual dataset logic to `DatasetToolbox`. This avoids duplicating business logic and keeps the MCP-specific code separate from the LangGraph agent package.

The server should not require `NEBIUS_API_KEY`, because it does not call the LLM or the LangGraph agent. It only loads the dataset and runs deterministic analysis tools.

## Runtime Behavior

The server is started manually from the command line:

```powershell
python -m bitext_mcp.server
```

By default, the server uses FastMCP's standard local transport behavior. The README client example will show a Python MCP client launching or connecting to the server and calling one tool, such as `get_dataset_schema` or `count_rows`.

Dataset path handling should default to `data/bitext_customer_support.csv`, matching the existing CLI. If the file is missing, `BitextDataset.from_path_or_download` may download it from Hugging Face, as it already does today.

## Tool Surface

Each MCP tool should have a clear docstring and typed parameters so client-side tool schemas are useful. The MCP tools should return the same JSON-serializable dictionaries as the existing `DatasetToolbox` methods.

Expected tools:

- `get_dataset_schema()`
  - Returns row count, columns, categories, and intents by category.
- `filter_dataset(category=None, intent=None, text_query=None, preview_rows=3)`
  - Creates a reusable filter and returns `filter_id`, count, criteria, and preview examples.
- `count_rows(filter_id=None, category=None, intent=None, text_query=None)`
  - Counts rows from either a previous filter or direct criteria.
- `show_examples(filter_id=None, category=None, intent=None, text_query=None, n=3, offset=0)`
  - Returns representative examples.
- `examples_by_category(n_per_category=1)`
  - Returns examples grouped by category.
- `intent_distribution(filter_id=None, category=None, text_query=None)`
  - Returns intent counts.
- `collect_response_patterns(category=None, intent=None, text_query=None, sample_size=8)`
  - Returns aggregate context and samples for response-pattern analysis.

## Error Handling

The server should preserve the existing deterministic validation behavior:

- Unknown categories and intents should surface as clear tool errors.
- Unknown `filter_id` values should tell the client to call `filter_dataset` first.
- Numeric bounds such as example count and preview count should remain constrained.

If FastMCP provides structured error responses for raised exceptions, the adapter can let `ValueError` propagate. If it is cleaner for MCP clients, the adapter may convert recoverable validation errors into JSON-shaped error payloads consistent with the existing LangChain tool wrapper.

## Testing

Preferred tests:

- Instantiate the FastMCP server through a project-owned `create_server(...)` factory.
- Use FastMCP's in-process client testing pattern if available.
- Call at least one exposed tool through the MCP client and verify the result.
- Verify that all expected tool names are registered.

Fallback tests if in-process client support is awkward in the installed FastMCP version:

- Test the server factory and adapter functions directly.
- Use a small in-memory `BitextDataset` fixture to avoid requiring network access.
- Verify delegation to `DatasetToolbox` behavior for schema, counts, examples, and distributions.

The normal test command remains:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Acceptance Criteria

- `python -m bitext_mcp.server` starts the MCP server.
- The MCP server exposes at least the seven dataset-analysis tools listed above.
- The MCP server can run without `NEBIUS_API_KEY`.
- README instructions are sufficient for a grader to start the MCP server and call one tool from a client.
- Existing CLI tests still pass.
- New MCP-related tests pass.
