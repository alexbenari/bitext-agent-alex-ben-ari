# Task 3 FastMCP Server ExecPlan

## Why this matters

The assignment requires the existing Bitext dataset tools to be available through MCP, so a grader or MCP-capable client can call the deterministic dataset analysis functions without using the interactive CLI or an LLM API key. This change adds a small FastMCP server package that reuses the current dataset and tool logic rather than duplicating it.

## Progress

- [x] (2026-05-25 18:35+03:00) Read assignment Task 3 and existing project structure.
- [x] (2026-05-25 18:55+03:00) Wrote and received approval for `docs/specs/task-3-mcp-server-spec.md`.
- [x] (2026-05-25 19:18+03:00) Confirm FastMCP import, server, and client APIs after dependency installation.
- [x] (2026-05-25 19:24+03:00) Add server tests.
- [x] (2026-05-25 19:26+03:00) Implement `bitext_mcp` server package.
- [x] (2026-05-25 19:18+03:00) Add dependency entries.
- [x] (2026-05-25 19:50+03:00) Update README with server startup and client call instructions.
- [x] (2026-05-25 19:52+03:00) Run full test suite and record outcome.

## Surprises & Discoveries

- Discovery: The current repo already has deterministic dataset tools in `bitext_agent.tools.DatasetToolbox`.
  Evidence: `bitext_agent/tools.py` defines `get_dataset_schema`, `filter_dataset`, `count_rows`, `show_examples`, `examples_by_category`, `intent_distribution`, and `collect_response_patterns`.
- Discovery: `fastmcp` is not installed in the current virtual environment.
  Evidence: probing `.venv` printed `fastmcp MISSING`.
- Discovery: The MCP server can avoid Nebius credentials because it only needs `BitextDataset` and `DatasetToolbox`.
  Evidence: `bitext_agent/dataset.py` and `bitext_agent/tools.py` do not depend on `AgentSettings`, `ChatOpenAI`, or `NEBIUS_API_KEY`.
- Discovery: FastMCP 2.14.7 supports in-process testing with `Client(server)`, and `call_tool` returns a result with `.data`.
  Evidence: local probe listed `ping` and returned `CallToolResult(..., data={'message': 'hello bitext'}, is_error=False)`.
- Discovery: MCP adapter tests pass through `fastmcp.Client`, including public argument schema checks and filter-id reuse across calls.
  Evidence: `.\.venv\Scripts\python.exe -m pytest tests\test_mcp_server.py -v` reported `5 passed`.
- Discovery: Running `python -m bitext_mcp.server` with redirected stdin starts FastMCP and then exits because stdio closes; the client launch path successfully starts the server and calls a tool.
  Evidence: background smoke showed the FastMCP banner; the README client example returned `{"count": 2992}`.
- Discovery: Git currently reports the project files as untracked, so `git diff` does not show the task diff.
  Evidence: `git status --short` lists repository files with `??`.
- Discovery: A standalone sample client is clearer for CLI use than an inline-only README snippet.
  Evidence: `.\.venv\Scripts\python.exe examples\mcp_client_example.py --category REFUND` returned clean JSON output.

## Decision Log

- Decision: Create a separate `bitext_mcp` package and keep `bitext_agent` unchanged except for shared imports.
  Rationale: This isolates the MCP surface while reusing existing dataset behavior. Moving dataset code into a new shared package would add refactor risk without improving Task 3 grading outcomes.
  Date/Author: 2026-05-25 / Codex
- Decision: Expose dataset-analysis tools only, not an `ask_agent` MCP tool.
  Rationale: Task 3 asks to expose tools. Avoiding the full LangGraph agent keeps MCP startup deterministic and free of LLM credentials.
  Date/Author: 2026-05-25 / Codex
- Decision: Expose all seven existing deterministic tools, not only the minimum three.
  Rationale: The existing methods are already JSON-serializable and useful to clients; exposing all of them improves completeness with little extra code.
  Date/Author: 2026-05-25 / Codex
- Decision: Keep `bitext_mcp.__init__` as a package marker instead of re-exporting `create_server`.
  Rationale: Re-exporting from `bitext_mcp.server` caused a `runpy` warning when starting the server with `python -m bitext_mcp.server`.
  Date/Author: 2026-05-25 / Codex

## Outcomes & Retrospective

Implemented Task 3 MCP support. The project now has a `bitext_mcp` package with a FastMCP server exposing seven deterministic dataset-analysis tools. The server reuses `BitextDataset` and `DatasetToolbox`, and it can be constructed without `NEBIUS_API_KEY`. A runnable sample client in `examples/mcp_client_example.py` launches the stdio MCP server, calls `count_rows`, and prints JSON.

Verification completed:

- `.\.venv\Scripts\python.exe -m pip install -e .` succeeded.
- FastMCP import and in-process client probe succeeded.
- `.\.venv\Scripts\python.exe -m pytest tests\test_mcp_server.py -v` reported `5 passed`.
- README client example returned `{"count": 2992}`.
- `.\.venv\Scripts\python.exe examples\mcp_client_example.py --category REFUND` returned `{"count": 2992}`.
- `.\.venv\Scripts\python.exe -m pytest` reported `64 passed` and one third-party deprecation warning from the test environment.
- `.\.venv\Scripts\python.exe -c "import os; os.environ.pop('NEBIUS_API_KEY', None); from bitext_mcp.server import create_server; server = create_server(); print(server is not None)"` printed `True`.

Residual risk: standalone FastMCP imports may emit a third-party deprecation warning from `authlib.jose`; this is external to project code and does not fail tests.

## Context and orientation

The project is a Python package for a Bitext customer-service dataset analyst agent. The CLI entry point is `main.py`. The LangGraph agent lives in `bitext_agent/graph.py`, and the command-line conversation loop lives in `bitext_agent/cli.py`.

The deterministic dataset logic already exists:

- `bitext_agent/dataset.py` loads the CSV, validates category and intent names, filters rows, and returns examples.
- `bitext_agent/tools.py` defines `DatasetToolbox`, which exposes dataset operations as regular Python methods and LangChain `StructuredTool` objects.
- `data/bitext_customer_support.csv` is the default local dataset cache. If missing, `BitextDataset.from_path_or_download` downloads it from Hugging Face.

Task 3 should add MCP support without changing CLI behavior. MCP means Model Context Protocol: a standard interface that lets tool clients discover and call server-defined tools. FastMCP is the Python package used to define and run the MCP server.

The approved feature spec is `docs/specs/task-3-mcp-server-spec.md`.

## Milestone 1 - Confirm FastMCP dependency and API shape

### Scope

Install or make available the FastMCP dependency, then verify the exact import and local client testing API before writing project code.

### Changes

- File: `pyproject.toml`
  Edit: add `fastmcp>=2.0.0,<3.0.0` to `[project].dependencies`.
- File: `requirements.txt`
  Edit: add `fastmcp>=2.0.0,<3.0.0`.

### Validation

- Command: `.\.venv\Scripts\python.exe -m pip install -e .`
  Expected: package installs successfully and includes `fastmcp`.
- Command: `.\.venv\Scripts\python.exe -c "from fastmcp import FastMCP, Client; print(FastMCP, Client)"`
  Expected: prints class objects without import errors.
- Command: run this temporary probe from PowerShell:

      @'
      import asyncio
      from fastmcp import Client, FastMCP

      mcp = FastMCP("Probe")

      @mcp.tool
      def ping(name: str = "world") -> dict[str, str]:
          return {"message": f"hello {name}"}

      async def main() -> None:
          async with Client(mcp) as client:
              tools = await client.list_tools()
              print([tool.name for tool in tools])
              result = await client.call_tool("ping", {"name": "bitext"})
              print(result)

      asyncio.run(main())
      '@ | .\.venv\Scripts\python.exe -

  Expected: output includes `ping` in the tool list and a result containing `hello bitext`.

### Rollback/Containment

If installation fails because the version range is wrong, inspect the installed package metadata and adjust only the FastMCP version bounds. Do not change `bitext_agent` code in this milestone.

## Milestone 2 - Add focused MCP server tests

### Scope

Add tests that define the expected MCP server behavior before implementation. The tests should avoid network access and LLM credentials.

### Changes

- File: `tests/test_mcp_server.py`
  Edit: create a new test module with:
  - A small in-memory `BitextDataset` fixture using `pandas.DataFrame`.
  - A test that `create_server(dataset)` registers all expected tool names.
  - A test that an in-process `fastmcp.Client` can call `count_rows`.
  - A test that no Nebius API key is required for server construction.

Suggested test structure:

      from __future__ import annotations

      import os

      import pandas as pd
      import pytest
      from fastmcp import Client

      from bitext_agent.dataset import BitextDataset
      from bitext_mcp.server import create_server


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


      @pytest.mark.asyncio
      async def test_mcp_server_registers_dataset_tools() -> None:
          server = create_server(make_dataset())

          async with Client(server) as client:
              tools = await client.list_tools()

          assert {tool.name for tool in tools} >= {
              "get_dataset_schema",
              "filter_dataset",
              "count_rows",
              "show_examples",
              "examples_by_category",
              "intent_distribution",
              "collect_response_patterns",
          }


      @pytest.mark.asyncio
      async def test_mcp_client_can_call_count_rows() -> None:
          server = create_server(make_dataset())

          async with Client(server) as client:
              result = await client.call_tool("count_rows", {"category": "REFUND"})

          assert result.data == {"count": 1}


      def test_create_server_does_not_require_nebius_key(monkeypatch) -> None:
          monkeypatch.delenv("NEBIUS_API_KEY", raising=False)

          server = create_server(make_dataset())

          assert server is not None

### Validation

- Command: `.\.venv\Scripts\python.exe -m pytest tests\test_mcp_server.py -v`
  Expected before implementation: fails because `bitext_mcp.server` does not exist.
  Expected after implementation: all tests pass.

### Rollback/Containment

If FastMCP's in-process `Client(server)` API behaves differently than the probe showed, change only the tests to use the confirmed FastMCP test pattern. Keep the expected product behavior the same.

## Milestone 3 - Implement the `bitext_mcp` package

### Scope

Create the server package and expose the seven dataset tools through FastMCP. Keep the package as a thin adapter over `DatasetToolbox`.

### Changes

- File: `bitext_mcp/__init__.py`
  Edit: create the package marker:

      """FastMCP server package for Bitext dataset tools."""

- File: `bitext_mcp/server.py`
  Edit: create the FastMCP server factory and command-line entry point. Use this structure:

      """FastMCP server exposing deterministic Bitext dataset tools."""

      from __future__ import annotations

      from pathlib import Path
      from typing import Any

      from fastmcp import FastMCP

      from bitext_agent.config import DEFAULT_DATASET_CACHE
      from bitext_agent.dataset import BitextDataset
      from bitext_agent.tools import DatasetToolbox


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

              return toolbox.filter_dataset(
                  category=category,
                  intent=intent,
                  text_query=text_query,
                  preview_rows=preview_rows,
              )

          @mcp.tool
          def count_rows(
              filter_id: str | None = None,
              category: str | None = None,
              intent: str | None = None,
              text_query: str | None = None,
          ) -> dict[str, Any]:
              """Count rows for a prior filter or direct filter criteria."""

              return toolbox.count_rows(
                  filter_id=filter_id,
                  category=category,
                  intent=intent,
                  text_query=text_query,
              )

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

              return toolbox.show_examples(
                  filter_id=filter_id,
                  category=category,
                  intent=intent,
                  text_query=text_query,
                  n=n,
                  offset=offset,
              )

          @mcp.tool
          def examples_by_category(n_per_category: int = 1) -> dict[str, Any]:
              """Return examples grouped by every dataset category."""

              return toolbox.examples_by_category(n_per_category=n_per_category)

          @mcp.tool
          def intent_distribution(
              filter_id: str | None = None,
              category: str | None = None,
              text_query: str | None = None,
          ) -> dict[str, Any]:
              """Return counts by intent for all rows, a category, text query, or prior filter."""

              return toolbox.intent_distribution(
                  filter_id=filter_id,
                  category=category,
                  text_query=text_query,
              )

          @mcp.tool
          def collect_response_patterns(
              category: str | None = None,
              intent: str | None = None,
              text_query: str | None = None,
              sample_size: int = 8,
          ) -> dict[str, Any]:
              """Collect rows and aggregates for qualitative response summaries."""

              return toolbox.collect_response_patterns(
                  category=category,
                  intent=intent,
                  text_query=text_query,
                  sample_size=sample_size,
              )

          return mcp


      def main() -> None:
          """Run the Bitext MCP server with FastMCP's default transport."""

          create_server().run()


      if __name__ == "__main__":
          main()

  Remove unused imports after implementation. If `Path` is unused, do not keep it.

### Validation

- Command: `.\.venv\Scripts\python.exe -m pytest tests\test_mcp_server.py -v`
  Expected: all MCP tests pass.
- Command: `.\.venv\Scripts\python.exe -m bitext_mcp.server`
  Expected: server starts and waits for MCP stdio input. Stop it with Ctrl+C after confirming startup.

### Rollback/Containment

If server startup blocks as expected, do not treat that as failure. If imports fail, inspect the installed FastMCP package and update `server.py` imports only.

## Milestone 4 - Update README documentation

### Scope

Document how a grader starts the MCP server and how a Python client calls one exposed tool.

### Changes

- File: `README.md`
  Edit: add `fastmcp` to setup expectations if dependency lists are mentioned.
  Edit: add a new section near "Run The CLI" or after "Tools" titled `Run The MCP Server`.

Content to include:

      ## Run The MCP Server

      Task 3 exposes the deterministic dataset-analysis tools through FastMCP. The MCP server does not call the LLM and does not require `NEBIUS_API_KEY`; it only needs the dataset CSV or network access for the first Hugging Face download.

      Start the server with:

          python -m bitext_mcp.server

      The server uses FastMCP's default local transport, so MCP clients can launch it as a Python process.

      Example client call:

          import asyncio
          from fastmcp import Client

          async def main() -> None:
              async with Client("bitext_mcp/server.py") as client:
                  result = await client.call_tool("count_rows", {"category": "REFUND"})
                  print(result.data)

          asyncio.run(main())

      Expected output shape:

          {"count": 2755}

      The exact count depends on the dataset version cached locally.

      Exposed MCP tools:

      - `get_dataset_schema`
      - `filter_dataset`
      - `count_rows`
      - `show_examples`
      - `examples_by_category`
      - `intent_distribution`
      - `collect_response_patterns`

### Validation

- Command: `Select-String -Path README.md -Pattern "Run The MCP Server","python -m bitext_mcp.server","call_tool"`
  Expected: all three patterns are found.

### Rollback/Containment

If the client example path needs to be a command object instead of a script path for the installed FastMCP version, update the README to match the verified client syntax from Milestone 1.

## Milestone 5 - Full verification

### Scope

Run the complete local verification suite and inspect changed files for accidental scope creep.

### Changes

- File: no planned code changes unless tests reveal a defect.

### Validation

- Command: `.\.venv\Scripts\python.exe -m pytest`
  Expected: all tests pass.
- Command: `git diff -- README.md pyproject.toml requirements.txt bitext_mcp tests docs`
  Expected: diff only includes Task 3 MCP server, tests, dependency docs, spec, and plan.
- Command: `.\.venv\Scripts\python.exe -c "import os; os.environ.pop('NEBIUS_API_KEY', None); from bitext_mcp.server import create_server; server = create_server(); print(server is not None)"`
  Expected: prints `True`.

### Rollback/Containment

If full tests fail outside MCP-related tests, inspect whether the failure is caused by this change. Do not modify unrelated graph, CLI, profile, prompt, or dataset behavior unless directly required by the MCP implementation.
