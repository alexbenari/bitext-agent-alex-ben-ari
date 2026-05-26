# Bitext Customer Service Data Analyst Agent

Task 1 implementation for `docs/assignment 3.pdf`: a LangGraph-based ReAct analyst agent for the Bitext customer-support dataset.

## Setup

Use Python 3.10 or newer. LangChain 1.x does not install on Python 3.9.

```powershell
python --version
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Set your Nebius Token Factory key before running the CLI agent:

```powershell
$env:NEBIUS_API_KEY = "your-token"
```

Optional overrides:

```powershell
$env:NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
$env:NEBIUS_MODEL = "nvidia/Nemotron-3-Nano-Omni"
```

## Run The CLI

```powershell
python main.py
```

To print CLI usage without starting the agent:

```powershell
python main.py --usage
```

The CLI prints a generated conversation session id after the welcome message:

```text
This conversation's session id is: session-... You can use it to resume it anytime after it ends.
```

To resume a saved conversation after exiting and restarting the app, pass that id with
`--session`:

```powershell
python main.py --session session-...
```

Long-term user profile memory is keyed separately from the conversation session. By
default the CLI uses the local `default` profile. To use a named profile:

```powershell
python main.py --user alex
```

The CLI prints the active profile at startup. When the agent notices durable user facts
or preferences after a successful dataset conversation, it proposes profile additions and
asks for approval before saving them. You can approve, discard, or edit the proposed
facts. Ask `What do you remember about me?` to inspect the current profile.

If the requested session id is not found, the CLI exits and asks you to verify the id or
start a new conversation without `--session`.

The first run downloads the CSV from Hugging Face into `data/bitext_customer_support.csv`. To use an already-downloaded file:

```powershell
python main.py --dataset-path data/bitext_customer_support.csv
```

The CLI prints the router decision, tool calls, tool observations, and final answer.

Each question also writes a structured JSONL run log under `logs/runs/`. These logs are intended for regression analysis and debugging; they include the run id, route decision, tool call arguments, tool result size, error type, latency, and token usage when the model provider returns it. The `logs/` directory is ignored by Git.

Conversation memory is persisted with LangGraph checkpoints in a local SQLite database
at `data/checkpoints.sqlite` by default. Override this path with:

```powershell
$env:BITEXT_CHECKPOINT_DB = "data/my_checkpoints.sqlite"
```

User profile facts are stored in a separate local SQLite database at
`data/user_profiles.sqlite` by default. Override this path with:

```powershell
$env:BITEXT_PROFILE_DB = "data/my_profiles.sqlite"
```

Checkpoint and profile database files are local runtime artifacts and are ignored by Git.

## Run The Streamlit UI

The bonus Streamlit UI runs the same LangGraph agent in a browser-based chat interface.
It requires `NEBIUS_API_KEY` because it calls the LLM-backed router and analyst graph.

From the project root:

```powershell
streamlit run streamlit_app.py
```

If Streamlit is not on your shell path, use the virtual environment executable:

```powershell
.\.venv\Scripts\streamlit.exe run streamlit_app.py
```

The sidebar shows the active conversation session id. You can paste an existing session
id from the Streamlit UI session store, choose a previous session from the recency-sorted
list, or use **Start new session** to create a fresh session. Unknown pasted session ids
are rejected so the UI only resumes sessions with a stored visible transcript.

The UI uses two local SQLite stores:

- `data/checkpoints.sqlite`: LangGraph checkpoint state used for agent continuation.
- `data/streamlit_sessions.sqlite`: Streamlit session metadata, previous-session list,
  visible chat turns, and trace events.

The UI displays the final answer in chat and keeps router, tool-call, observation, and
error events under each assistant message's run-details expander. The raw trace JSON is
available from the same panel for debugging.

## Run The MCP Server

Task 3 exposes the deterministic dataset-analysis tools through FastMCP. The MCP server
does not call the LLM and does not require `NEBIUS_API_KEY`; it only needs the dataset
CSV or network access for the first Hugging Face download.

### Start The Server

From the project root, start the MCP server with:

```powershell
python -m bitext_mcp.server
```

The server uses FastMCP's default local transport, so MCP clients can launch it as a
Python process.

### Connect A Client In Code

For stdio-based MCP, the client usually launches the server process and talks to it over
stdin/stdout. This example connects to `bitext_mcp/server.py` and calls `count_rows`:

```python
import asyncio
import json

from fastmcp import Client


async def main() -> None:
    async with Client("bitext_mcp/server.py") as client:
        result = await client.call_tool("count_rows", {"category": "REFUND"})
        print(json.dumps(result.data))


asyncio.run(main())
```

Expected output with the bundled dataset:

```text
{"count": 2992}
```

### Run The Sample Client

The repo includes a runnable sample client that starts the MCP server, confirms the
connection, calls the MCP client's `list_tools()` method, prints the available tools,
calls `get_dataset_schema`, prints the dataset categories, and exits.

Run it from the project root:

```powershell
python -m examples.mcp_client_example
```

Exposed MCP tools:

- `get_dataset_schema`
- `filter_dataset`
- `count_rows`
- `show_examples`
- `examples_by_category`
- `intent_distribution`
- `collect_response_patterns`

## Architecture

The application uses a LangGraph `StateGraph`:

1. `router`: classifies each question as `structured`, `unstructured`, or `out_of_scope` using `nvidia/Nemotron-3-Nano-Omni`.
2. `decline`: politely rejects out-of-scope questions without using general model knowledge.
3. `agent`: a LangChain `create_agent` ReAct loop, also using `nvidia/Nemotron-3-Nano-Omni`.

The compiled graph runs with a SQLite checkpointer, keyed by the CLI `--session` value.
When no session is passed, the CLI creates a unique `session-<uuid>` id for that run and
prints it so the user can resume the conversation later.

Profile facts are stored separately from conversation checkpoints in SQLite tables keyed
by `--user`. The router sends profile-inspection questions to a deterministic profile
node. Successful dataset conversations flow through a profile extraction node using
`Qwen/Qwen3.5-397B-A17B`, which proposes facts for user approval before they are saved.

The same Nebius Token Factory model is used for routing and generation to match the assignment requirement. Nebius Token Factory is OpenAI-compatible, so the code uses `ChatOpenAI` with `NEBIUS_BASE_URL`.

## Tools

The agent has deterministic tools over the dataset, each with a Pydantic argument schema:

- `get_dataset_schema`: row count, columns, categories, and intents.
- `filter_dataset`: create a reusable filter over category, intent, and text query.
- `count_rows`: count all rows or rows from a filter.
- `show_examples`: show instructions and responses from all rows or a filter.
- `examples_by_category`: show examples grouped by every category.
- `intent_distribution`: count intents overall, in a category, or within a filter.
- `collect_response_patterns`: gather response samples and aggregate context for open-ended summaries.

## Prompts

The two reviewable prompts live as Markdown files under `prompts/`:

- `prompts/router.md`
- `prompts/agent_system.md`

Both include the dataset category and intent taxonomy from the Bitext dataset card.

## Example Queries

```text
What categories exist in the dataset?
How many refund requests did we get?
Show me 5 examples of the SHIPPING category.
Summarize how agents respond to complaint intents.
Show me examples of people wanting their money back.
What is the distribution of intents in the ACCOUNT category?
What's the best CRM software for handling complaints?
Who is the president of France?
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```
