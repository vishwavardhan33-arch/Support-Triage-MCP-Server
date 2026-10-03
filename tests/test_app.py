import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

import app as app_module  # noqa: E402
import server  # noqa: E402

client = TestClient(app_module.app)


def test_health_reports_ollama_status():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert "ollama" in r.json()
    assert "model" in r.json()


def test_list_tickets():
    r = client.get("/api/tickets")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
    assert len(r.json()) > 0


def test_list_tickets_filtered_by_status():
    r = client.get("/api/tickets?status=open")
    assert r.status_code == 200
    assert all(t["status"] == "open" for t in r.json())


def test_get_ticket_found():
    all_tickets = client.get("/api/tickets").json()
    tid = all_tickets[0]["id"]
    r = client.get(f"/api/tickets/{tid}")
    assert r.status_code == 200
    assert r.json()["id"] == tid


def test_get_ticket_not_found():
    r = client.get("/api/tickets/NOPE")
    assert r.status_code == 404


def test_update_status_roundtrip():
    all_tickets = client.get("/api/tickets").json()
    tid = all_tickets[0]["id"]
    original = all_tickets[0]["status"]

    r = client.patch(f"/api/tickets/{tid}/status", json={"status": "closed"})
    assert r.status_code == 200
    assert r.json()["status"] == "closed"

    client.patch(f"/api/tickets/{tid}/status", json={"status": original})


def test_classify_endpoint_mocked():
    fake_json = (
        '{"category": "Disbursement Delay", "priority": "high", '
        '"sentiment": "frustrated", "suggested_response": "We are escalating this now."}'
    )
    fake_resp = MagicMock()
    fake_resp.raise_for_status.return_value = None
    fake_resp.json.return_value = {"response": fake_json}

    all_tickets = client.get("/api/tickets").json()
    tid = all_tickets[0]["id"]

    with patch.object(server.requests, "post", return_value=fake_resp):
        r = client.post(f"/api/tickets/{tid}/classify")

    assert r.status_code == 200
    assert r.json()["category"] == "Disbursement Delay"


def test_classify_endpoint_ollama_down():
    import requests as requests_module

    all_tickets = client.get("/api/tickets").json()
    tid = all_tickets[0]["id"]

    with patch.object(server.requests, "post", side_effect=requests_module.exceptions.ConnectionError()):
        r = client.post(f"/api/tickets/{tid}/classify")

    assert r.status_code == 502


def test_frontend_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "Support triage" in r.text
