# Support Triage MCP
[![M8ven Score](https://m8ven.ai/badge/mcp/vishwavardhan33-arch/support-triage-mcp-server)](https://m8ven.ai/mcp/vishwavardhan33-arch/support-triage-mcp-server?s=readme)
An MCP (Model Context Protocol) server that exposes a support-ticket inbox as
tools an AI client can use to list, inspect, triage, and update tickets.
The dataset here is mocked (styled after a home-loan partner-support inbox:
loan status queries, document issues, disbursement delays, KYC, etc.) but the
server is written so a real ticket source (Zendesk, email, a helpdesk API)
could be swapped in later without changing the tool interface.

## Status

Core plumbing (`list_tickets`, `get_ticket`, `update_ticket_status`) and
LLM-based triage (`classify_ticket`, with RAG-style retrieval of similar
past tickets) are both working. `classify_ticket` runs entirely on a local,
open-source LLM via [Ollama](https://ollama.com) — no API key, no per-call
cost. See [Roadmap](#roadmap) for what's next.

### Setting up Ollama (needed for `classify_ticket`)

1. Install Ollama from [ollama.com](https://ollama.com) (Windows/Mac/Linux).
2. Pull a model (one-time download, then it runs fully offline):
   ```bash
   ollama pull llama3.2
   ```
   `llama3.2` (3B) is small and fast enough for CPU-only laptops. For
   better classification quality if your machine can handle it, try
   `ollama pull llama3.1` (8B) or `ollama pull mistral` — just set
   `OLLAMA_MODEL` to match (see below).
3. Ollama runs a local server automatically after install. If it's not
   running, start it with:
   ```bash
   ollama serve
   ```

By default the server talks to `http://localhost:11434` and uses the
`llama3.2` model. Override either with environment variables if needed:

```bash
# macOS/Linux
export OLLAMA_MODEL="mistral"

# Windows (PowerShell)
$env:OLLAMA_MODEL="mistral"
```

`list_tickets`, `get_ticket`, and `update_ticket_status` don't need Ollama
at all — only `classify_ticket` does.

## Evaluation

`evaluate.py` measures two things against the mock dataset:

1. **Routing accuracy** — how often `classify_ticket`'s predicted `category`
   and `priority` match the dataset's ground-truth labels (baked in by
   `generate_data.py`).
2. **LLM-as-judge** — there's no ground truth for the *quality* of the
   drafted `suggested_response`, so a second local model scores it on
   relevance, professionalism, and actionability (1-5 each).

```bash
# Quick run: 10 tickets, routing accuracy + judge scores
python evaluate.py --limit 10

# Use a different, larger model as judge to reduce self-judging bias
python evaluate.py --limit 10 --judge-model mistral

# Routing accuracy only, no judge step (faster)
python evaluate.py --limit 10 --no-judge
```

Each ticket costs 1-2 local LLM calls, so on a CPU-only laptop start with
`--limit 5-10` before running the full 40-ticket dataset — otherwise it can
take a while (see [Status](#status) for rough per-call timing). Full
per-ticket results are written to `eval_results.json` (git-ignored, since
it's a run artifact, not source).

**Note on self-judging bias:** if the same model both classifies and judges,
it tends to rate its own drafts generously. Passing `--judge-model` with a
different (ideally larger) model gives a more honest signal.

## Web UI

On top of the MCP server, `app.py` wraps the same tool functions as a small
FastAPI REST API and serves a browser-based dashboard from `static/` — so
the project looks and works like a real internal support tool, not just a
script.

```bash
python app.py
```

Then open **http://localhost:8000**. Filter tickets by status/category in
the left rail, click a case to open its detail panel, update its status, or
click **Run AI triage** to call `classify_ticket` (via Ollama) and see the
predicted category, priority, sentiment, and a copyable drafted response.
The dot next to the header shows whether Ollama is currently reachable.

This is a separate, optional layer — the MCP server and its tools work
identically with or without the web UI running.

## Deploying the Web UI

`app.py` + `static/` is a self-contained FastAPI app (API + dashboard in
one), which makes it deployable as a single web service on platforms like
[Render](https://render.com) or [Fly.io](https://fly.io) — no separate
frontend host needed.

Since a deployed server has no local Ollama to talk to, point it at a
hosted, OpenAI-compatible LLM API instead — [Groq](https://console.groq.com)
has a free tier (no card required) and is the easiest drop-in, since
`server.py`'s `_call_llm()` already branches between local Ollama and Groq
based on `OLLAMA_URL` (see `.env.example`).

**Steps (Render):**

1. Get a free API key at [console.groq.com](https://console.groq.com).
2. Push this repo to GitHub.
3. On Render: **New → Web Service**, connect the repo. If you keep
   `render.yaml` in the repo root, Render will read the build/start
   commands and env vars from it automatically (Blueprint deploy) — you'll
   just need to paste in your `GROQ_API_KEY` when prompted, since it's
   deliberately left out of the committed config.
4. Otherwise set manually:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python app.py`
   - **Environment variables:**
     - `OLLAMA_URL=https://api.groq.com/openai/v1/chat/completions`
     - `OLLAMA_MODEL=llama-3.1-8b-instant`
     - `GROQ_API_KEY=<your key>`
5. Once deployed, open the Render-provided URL — you should see the
   dashboard, and "Run AI triage" will call Groq instead of local Ollama.

The MCP server (`server.py` run standalone, or via Claude Desktop) isn't
something you deploy to a URL — it speaks MCP over stdio to a client, not
HTTP to a browser. Demo that part locally or via screen recording instead.

## Project structure

```
support-triage-mcp/
├── .github/
│   └── workflows/
│       └── ci.yml         # GitHub Actions: lint + pytest on every push/PR
├── app.py                 # FastAPI backend for the web UI (wraps the same tool functions)
├── static/                # Web UI frontend (HTML/CSS/JS, served by app.py)
├── server.py              # MCP server + tool definitions
├── evaluate.py            # routing-accuracy + LLM-as-judge evaluation harness
├── retrieval.py           # TF-IDF similar-ticket retrieval (RAG-style)
├── data/
│   ├── tickets.json       # mock ticket dataset
│   └── generate_data.py   # regenerates tickets.json
├── tests/
│   └── test_server.py     # smoke tests for the tool functions
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd support-triage-mcp
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Regenerate the mock dataset any time with:

```bash
python data/generate_data.py
```

Run the test suite:

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Lint (same check CI runs):

```bash
ruff check .
```

## Running the server standalone

```bash
python server.py
```

This starts the MCP server over stdio, waiting for a client to connect.

## Connecting to Claude Desktop

Add this to your Claude Desktop MCP config
(`claude_desktop_config.json` — Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "support-triage": {
      "command": "python",
      "args": ["/absolute/path/to/support-triage-mcp/server.py"]
    }
  }
}
```

Restart Claude Desktop, then try prompts like:

- "List all open support tickets"
- "Show me ticket T-1021"
- "Mark T-1021 as in_progress"
- "Classify ticket T-1021 and draft a first response"

## Available tools

| Tool | Description |
|---|---|
| `list_tickets(status?, category?)` | List tickets, optionally filtered by status (`open` / `in_progress` / `closed`) and/or category. |
| `get_ticket(ticket_id)` | Get full details of one ticket by ID. |
| `update_ticket_status(ticket_id, status)` | Update a ticket's status. |
| `classify_ticket(ticket_id)` | Category/priority/sentiment classification + a suggested first response, using a local open-source LLM (Ollama) and similar past tickets as reference context. |

## Roadmap

- [ ] Swap TF-IDF retrieval for real embeddings (a local embedding model, e.g. via `ollama pull nomic-embed-text`)
- [ ] Swap the mock dataset for a real ticket source
- [ ] Try larger local models (`llama3.1`, `mistral`) and compare routing accuracy vs. `llama3.2`

## Continuous Integration

Every push and PR to `main` runs `.github/workflows/ci.yml`, which lints
with [ruff](https://docs.astral.sh/ruff/) and runs the full pytest suite on
Python 3.11 and 3.12. No Ollama install is needed in CI — `classify_ticket`'s
LLM call is mocked in `tests/test_classify.py`, so the suite runs fully
offline. You'll see a status badge/checkmark on each commit and PR once this
is pushed.

## Notes on the MCP SDK version

This project uses `mcp` v2.x, where the server class was renamed from
`FastMCP` to `MCPServer` (`mcp.server.mcpserver.MCPServer`). The decorator
API (`@mcp.tool()`) and `mcp.run()` are unchanged from v1. If you're
following older MCP tutorials that import `from mcp.server.fastmcp import
FastMCP`, either update the import as above or pin `mcp<2.0.0`.
