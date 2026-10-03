"""
Basic smoke tests for the support-triage MCP server's tool functions.
Run: pytest tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from server import list_tickets, get_ticket, update_ticket_status  # noqa: E402


def _call(tool, *args, **kwargs):
    """Tool functions decorated with @mcp.tool() are wrapped; .fn gives the raw function."""
    fn = getattr(tool, "fn", tool)
    return fn(*args, **kwargs)


def test_list_tickets_returns_data():
    tickets = _call(list_tickets)
    assert isinstance(tickets, list)
    assert len(tickets) > 0


def test_list_tickets_filters_by_status():
    open_tickets = _call(list_tickets, status="open")
    assert all(t["status"] == "open" for t in open_tickets)


def test_list_tickets_rejects_bad_status():
    result = _call(list_tickets, status="not_a_status")
    assert "error" in result[0]


def test_get_ticket_found():
    all_tickets = _call(list_tickets)
    first_id = all_tickets[0]["id"]
    ticket = _call(get_ticket, first_id)
    assert ticket["id"] == first_id


def test_get_ticket_not_found():
    result = _call(get_ticket, "T-9999999")
    assert "error" in result


def test_update_ticket_status_roundtrip():
    all_tickets = _call(list_tickets)
    ticket_id = all_tickets[0]["id"]
    original_status = all_tickets[0]["status"]

    updated = _call(update_ticket_status, ticket_id, "closed")
    assert updated["status"] == "closed"

    # restore original status so repeated test runs stay deterministic
    _call(update_ticket_status, ticket_id, original_status)
