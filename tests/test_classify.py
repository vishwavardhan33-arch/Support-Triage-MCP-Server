import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

import server  # noqa: E402


def _call(tool, *args, **kwargs):
    fn = getattr(tool, "fn", tool)
    return fn(*args, **kwargs)


def _fake_ollama_response(json_text: str):
    """Builds a fake `requests.post` response mimicking Ollama's /api/generate."""
    fake = MagicMock()
    fake.raise_for_status.return_value = None
    fake.json.return_value = {"response": json_text}
    return fake


def test_classify_ticket_ollama_not_running():
    all_tickets = _call(server.list_tickets)
    ticket_id = all_tickets[0]["id"]

    with patch.object(server.requests, "post", side_effect=requests.exceptions.ConnectionError()):
        result = _call(server.classify_ticket, ticket_id)

    assert "error" in result
    assert "Ollama" in result["error"]


def test_classify_ticket_not_found():
    with patch.object(server.requests, "post") as mock_post:
        result = _call(server.classify_ticket, "T-9999999")
    mock_post.assert_not_called()  # should fail fast before ever calling the LLM
    assert "error" in result


def test_classify_ticket_parses_llm_json():
    fake_json = (
        '{"category": "Disbursement Delay", "priority": "high", '
        '"sentiment": "frustrated", "suggested_response": "We are escalating this now."}'
    )

    all_tickets = _call(server.list_tickets)
    ticket_id = all_tickets[0]["id"]

    with patch.object(server.requests, "post", return_value=_fake_ollama_response(fake_json)):
        result = _call(server.classify_ticket, ticket_id)

    assert result["category"] == "Disbursement Delay"
    assert result["priority"] == "high"
    assert result["ticket_id"] == ticket_id
    assert "similar_ticket_ids" in result


def test_classify_ticket_handles_bad_json():
    with patch.object(server.requests, "post", return_value=_fake_ollama_response("not valid json")):
        all_tickets = _call(server.list_tickets)
        ticket_id = all_tickets[0]["id"]
        result = _call(server.classify_ticket, ticket_id)

    assert "error" in result
    assert "raw_output" in result
