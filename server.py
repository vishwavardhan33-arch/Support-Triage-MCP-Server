"""
Support Triage MCP Server
--------------------------
Exposes support-ticket data (currently a local mock dataset standing in for
a Zendesk/email inbox) as MCP tools so an MCP-compatible client (e.g. Claude
Desktop) can list, inspect, and update tickets.

Tools:
    - list_tickets(status=None, category=None)
    - get_ticket(ticket_id)
    - update_ticket_status(ticket_id, status)
    - classify_ticket(ticket_id)  -- LLM-based triage via a local, open-source
      model served by Ollama (free, runs on your own machine, no API cost).
"""

import json
import os
from pathlib import Path
from typing import Optional

import requests
from mcp.server.mcpserver import MCPServer

from retrieval import find_similar_tickets

DATA_PATH = Path(__file__).parent / "data" / "tickets.json"
VALID_STATUSES = {"open", "in_progress", "closed"}
CATEGORIES = [
    "Loan Status Query", "Document Upload Issue", "Disbursement Delay",
    "Interest Rate Query", "Account Access", "Partner Onboarding",
    "KYC Issue", "Payment Reconciliation", "Technical / Portal Bug",
    "Complaint - Escalation",
]

# By default this talks to a locally-running Ollama instance (free, no API
# key, runs on your own machine). Install: https://ollama.com | Pull a
# model once: `ollama pull llama3.2`
#
# For deployment (e.g. Render, Fly.io) there's usually no local Ollama to
# talk to, so this also supports pointing at Groq's hosted, OpenAI-compatible
# chat API instead — set OLLAMA_URL to Groq's endpoint and provide
# GROQ_API_KEY. The two APIs have different request/response shapes, so
# _call_llm() below branches on which one OLLAMA_URL points at.
#
#   Local (default):
#     OLLAMA_URL=http://localhost:11434/api/generate
#     OLLAMA_MODEL=llama3.2
#
#   Hosted (Groq, free tier available at console.groq.com):
#     OLLAMA_URL=https://api.groq.com/openai/v1/chat/completions
#     OLLAMA_MODEL=openai/gpt-oss-20b
#     GROQ_API_KEY=<your key>
#
#   NOTE: Groq periodically deprecates/renames model IDs (it happened to
#   llama-3.1-8b-instant in Aug 2026). A 404 from the Groq branch below
#   usually means OLLAMA_MODEL is stale — check current IDs with:
#     curl -s https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
USING_GROQ = "groq.com" in OLLAMA_URL

CLASSIFY_SYSTEM_PROMPT = f"""You are a support-ticket triage assistant for a home loan \
partner-support team. Given a new ticket and some similar past tickets (for reference \
only), classify the new ticket and draft a short first response.

Respond with ONLY a JSON object, no other text, no markdown fences, in this exact shape:
{{
  "category": "<one of: {', '.join(CATEGORIES)}>",
  "priority": "<low|medium|high>",
  "sentiment": "<neutral|frustrated|angry|satisfied>",
  "suggested_response": "<a short, professional 2-3 sentence first response to the sender>"
}}"""

mcp = MCPServer("Support Triage")


def _call_groq(system: str, user_prompt: str, model: str) -> str:
    """
    Sends a prompt to Groq's hosted, OpenAI-compatible chat completions API
    and returns the raw text response. Requires GROQ_API_KEY to be set —
    get a free key at console.groq.com.
    """
    if not GROQ_API_KEY:
        raise RuntimeError(
            "OLLAMA_URL points at Groq but GROQ_API_KEY is not set. "
            "Get a free key at https://console.groq.com and set it as an "
            "environment variable."
        )
    try:
        response = requests.post(
            OLLAMA_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"Could not connect to Groq at {OLLAMA_URL}.") from e
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Groq API error: {e}") from e

    return response.json()["choices"][0]["message"]["content"]


def _call_ollama(system: str, user_prompt: str, model: str) -> str:
    """
    Sends a prompt to a locally-running Ollama instance and returns the raw
    text response. Requires `ollama serve` running and the model pulled
    (`ollama pull <model>`) — see README for setup.
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "system": system,
                "prompt": user_prompt,
                "stream": False,
                "format": "json",  # asks Ollama to constrain output to valid JSON
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Could not connect to Ollama at {OLLAMA_URL}. "
            f"Make sure it's installed and running (`ollama serve`), and that you've "
            f"pulled the model with `ollama pull {model}`."
        ) from e

    return response.json()["response"]


def _call_llm(system: str, user_prompt: str, model: Optional[str] = None) -> str:
    """
    Dispatches to Groq or local Ollama depending on OLLAMA_URL, so the rest
    of the code (classify_ticket, tests) doesn't need to know which one is
    in play.
    """
    model = model or OLLAMA_MODEL
    if USING_GROQ:
        return _call_groq(system, user_prompt, model)
    return _call_ollama(system, user_prompt, model)


def _build_classify_prompt(ticket: dict, similar: list[dict]) -> str:
    lines = [
        "NEW TICKET:",
        f"Subject: {ticket['subject']}",
        f"Body: {ticket['body']}",
        f"Sender type: {ticket['sender_type']}",
    ]
    if similar:
        lines.append("\nSIMILAR PAST TICKETS (reference only, do not copy verbatim):")
        for t in similar:
            lines.append(f"- Subject: {t['subject']} | Category: {t['category']} | Priority: {t['priority']}")
    return "\n".join(lines)


def _load_tickets() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        return json.load(f)


def _save_tickets(tickets: list[dict]) -> None:
    with open(DATA_PATH, "w") as f:
        json.dump(tickets, f, indent=2)


@mcp.tool()
def list_tickets(status: Optional[str] = None, category: Optional[str] = None) -> list[dict]:
    """
    List support tickets, optionally filtered.

    Args:
        status: filter by ticket status - "open", "in_progress", or "closed".
        category: filter by ticket category, e.g. "Disbursement Delay".
    """
    tickets = _load_tickets()

    if status:
        if status not in VALID_STATUSES:
            return [{"error": f"Invalid status '{status}'. Must be one of {sorted(VALID_STATUSES)}"}]
        tickets = [t for t in tickets if t["status"] == status]

    if category:
        tickets = [t for t in tickets if t["category"].lower() == category.lower()]

    return tickets


@mcp.tool()
def get_ticket(ticket_id: str) -> dict:
    """
    Get full details of a single ticket by its ID (e.g. "T-1021").
    """
    tickets = _load_tickets()
    for t in tickets:
        if t["id"] == ticket_id:
            return t
    return {"error": f"Ticket '{ticket_id}' not found"}


@mcp.tool()
def update_ticket_status(ticket_id: str, status: str) -> dict:
    """
    Update a ticket's status.

    Args:
        ticket_id: the ticket to update, e.g. "T-1021".
        status: new status - "open", "in_progress", or "closed".
    """
    if status not in VALID_STATUSES:
        return {"error": f"Invalid status '{status}'. Must be one of {sorted(VALID_STATUSES)}"}

    tickets = _load_tickets()
    for t in tickets:
        if t["id"] == ticket_id:
            t["status"] = status
            _save_tickets(tickets)
            return t

    return {"error": f"Ticket '{ticket_id}' not found"}


@mcp.tool()
def classify_ticket(ticket_id: str) -> dict:
    """
    Classify a ticket using an open-source LLM (local via Ollama, or a
    hosted provider like Groq if configured — see OLLAMA_URL above):
    predicts category, priority, and sentiment, and drafts a suggested
    first response. Retrieves similar past (non-open) tickets as reference
    context before classifying (RAG-style retrieval, see retrieval.py).
    """
    tickets = _load_tickets()
    ticket = next((t for t in tickets if t["id"] == ticket_id), None)
    if ticket is None:
        return {"error": f"Ticket '{ticket_id}' not found"}

    similar = find_similar_tickets(ticket, tickets, top_k=3)
    user_prompt = _build_classify_prompt(ticket, similar)

    try:
        raw_text = _call_llm(CLASSIFY_SYSTEM_PROMPT, user_prompt)
    except RuntimeError as e:
        return {"error": str(e)}

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned[4:] if cleaned.lower().startswith("json") else cleaned

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        return {"error": "Could not parse model output as JSON", "raw_output": raw_text}

    result["ticket_id"] = ticket_id
    result["similar_ticket_ids"] = [t["id"] for t in similar]
    return result


if __name__ == "__main__":
    mcp.run()
