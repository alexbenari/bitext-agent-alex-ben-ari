# Bitext Customer Service Data Analyst Agent

A LangGraph-based ReAct analyst agent for the Bitext customer-support dataset, with persistent conversation and user-profile memory, a FastMCP tool server, and Streamlit integration.

## Setup

### Initial Setup

These commands create a clean environment, install pinned project dependencies, set the
required Nebius Token Factory key, and start the CLI agent:

Use Python 3.10, 3.11, 3.12, or 3.13. Python 3.11 was used for development and testing.
Python 3.14 is not currently supported by the dependency stack. The commands below use
Python 3.11; if you already have another supported version installed, replace `py -3.11`
with that version, for example `py -3.12`.

```powershell
git clone https://github.com/alexbenari/bitext-agent-alex-ben-ari.git
cd bitext-agent-alex-ben-ari
py -0p
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
$env:NEBIUS_API_KEY = "your-token"
python main.py
```

By default, the app loads the bundled dataset at `data/bitext_customer_support.csv`.
Because this file is checked into the repo, normal runs do not need a Hugging Face
download.

If the CSV is missing, the app downloads it from Hugging Face and writes it to the
configured dataset path. To point the app at another local copy, use:

```powershell
python main.py --dataset-path path\to\bitext_customer_support.csv
```

To print CLI usage without starting the agent:

```powershell
python main.py --usage
```

### Agent interaction
When started the agent displays a welcome message and waits for instructions.
The agent's replies are written to the CLI in white and tool calls, reasoning and graph outputs are written to the CLI in grey to differentiate them from the actual conversation.

### Session Management

Each conversation is identified and saved using a unique session id generated at the beginning of the conversation. The CLI prints the conversation session id after the agent's welcome message.

To resume a saved conversation after exiting and restarting the app, pass that id with
`--session`:

```powershell
python main.py --session session-...
```
If the requested session id is not found, the CLI exits and asks the user to verify the id or
start a new conversation without `--session`.

Conversation memory is persisted with LangGraph checkpoints in a local SQLite database
at `data/checkpoints.sqlite` by default. Override this path with:

```powershell
$env:BITEXT_CHECKPOINT_DB = "data/my_checkpoints.sqlite"
```

### User Profile

Long-term user profile memory is saved separately from the conversation session, keyed by username.
By default, the CLI uses the `default` profile. To use a named profile, pass `--user`:

```powershell
python main.py --user alex
```

The CLI prints the active profile at startup. When the agent notices durable user facts
or preferences after a successful dataset conversation, it proposes profile additions and
asks for approval before saving them. The user can approve, discard, or edit the proposed
facts.

A user can also ask to view their profile using phrasing like `What do you remember about me?`, `show me my user profile`, and similar requests.

User profile facts are stored in a separate local SQLite database at
`data/user_profiles.sqlite` by default. Override this path with:

```powershell
$env:BITEXT_PROFILE_DB = "data/my_profiles.sqlite"
```

### CLI Output and Logs

The CLI prints the router decision, tool calls, tool observations, and final answer.

Each question also writes a structured JSONL run log under `logs/runs/`. These logs are intended for regression analysis and debugging; they include the run id, route decision, tool call arguments, tool result size, error type, latency, and token usage when the model provider returns it.

## Streamlit UI

From the project root:

```powershell
streamlit run streamlit_app.py
```

If Streamlit is not on the shell path, use the virtual environment executable:

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

## MCP Server

From the project root, start the MCP server with:

```powershell
python -m bitext_mcp.server
```

The server uses FastMCP's default local transport, so MCP clients can launch it as a
Python process.

Exposed MCP tools:

- `get_dataset_schema`
- `filter_dataset`
- `count_rows`
- `show_examples`
- `examples_by_category`
- `intent_distribution`
- `collect_response_patterns`

### Connect a Client in Code

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

### Run the Sample Client

The repo includes a runnable sample client that starts the MCP server, confirms the
connection, calls the MCP client's `list_tools()` method, prints the available tools,
calls `get_dataset_schema`, prints the dataset categories, and exits.

Run it from the project root:

```powershell
python -m examples.mcp_client_example
```

## Model Choice

- The agent uses `nvidia/Nemotron-3-Nano-Omni` for both routing and ReAct analysis. I chose the same model for both roles for simplicity. `nvidia/Nemotron-3-Nano-Omni` is a reasoning model with tool calling, which is necessary for the ReAct analyst but not strictly required for the router. A smaller model could probably handle routing, but routing queries are short, this model is inexpensive, and price and latency differences are not significant in this exercise. If this were a production agent with significant usage, I would optimize the router to use a simpler model.
- Profile extraction uses `Qwen/Qwen3.5-397B-A17B`. This task is mostly conservative information extraction and structured output, not deep multi-step reasoning. I chose a strong instruction-following text model that can avoid over-inferring personal facts. In this exercise, latency is not a major concern. In production, I would explore faster variants such as `Qwen/Qwen3.5-397B-A17B-fast`.

## Architecture

### Graph structure
The application uses a LangGraph `StateGraph` with these nodes:

1. `router`: classifies each question as `structured`, `unstructured`, `profile_info`, or `out_of_scope`.
2. `decline`: politely rejects out-of-scope questions without answering from the LLM's general knowledge.
3. `profile_info`: answers profile-inspection questions, such as `What do you remember about me?`, from the persistent user-profile store.
4. `agent`: runs a LangChain `create_agent` ReAct loop with the dataset tools.
5. `profile_extractor`: reviews successful conversations and proposes durable user-profile facts extracted from them.

### Graph flow
Every question enters through `router`. Out-of-scope questions go to `decline` and then end. Profile-inspection questions go to `profile_info` and then end. Structured and unstructured dataset questions go to `agent`, which can call one or more deterministic dataset tools before producing a final answer. Successful dataset turns then pass through `profile_extractor`; any proposed profile facts are returned to the CLI or UI so the user can approve, edit, or discard them before they are saved. The compiled graph runs with a SQLite checkpointer keyed by the CLI `--session` value, so the same session id resumes the same conversation after a restart. Profile facts are stored separately in SQLite tables keyed by `--user`.

If the router model fails or returns invalid structured output, the app returns a technical-problem fallback instead of guessing a route. If the ReAct loop exceeds the configured iteration limit, the agent returns a graceful max-iteration fallback. If the agent fails during live execution, the app emits an error trace event, writes the error status to the JSONL run log when logging is enabled, and returns the same technical-problem fallback. If profile extraction fails, the dataset answer is preserved and the app simply returns no proposed profile facts for that turn.

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

The three reviewable prompts live as Markdown files under `prompts/`:

- `prompts/router.md`
- `prompts/agent_system.md`
- `prompts/profile_extractor.md`

The router and agent prompts include the dataset category and intent taxonomy from the Bitext dataset card. The profile-extractor prompt defines the conservative rules for proposing durable user-profile facts.
