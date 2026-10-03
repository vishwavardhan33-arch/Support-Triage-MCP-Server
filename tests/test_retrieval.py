import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval import find_similar_tickets  # noqa: E402

TICKETS = [
    {"id": "T-1", "subject": "Disbursement Delay: BHL-1", "body": "Funds not credited yet for BHL-1", "status": "closed"},
    {"id": "T-2", "subject": "Disbursement Delay: BHL-2", "body": "Sanctioned amount for BHL-2 not credited", "status": "closed"},
    {"id": "T-3", "subject": "Interest Rate Query: BHL-3",
     "body": "Customer wants clarification on rate slab", "status": "closed"},
    {"id": "T-4", "subject": "Disbursement Delay: BHL-4", "body": "Disbursement for BHL-4 delayed", "status": "open"},
]


def test_finds_textually_similar_tickets():
    target = {"id": "T-99", "subject": "Disbursement Delay: BHL-99", "body": "Funds for BHL-99 haven't arrived"}
    results = find_similar_tickets(target, TICKETS, top_k=2)
    result_ids = [t["id"] for t in results]
    assert "T-1" in result_ids or "T-2" in result_ids


def test_excludes_open_tickets():
    target = {"id": "T-99", "subject": "Disbursement Delay: BHL-99", "body": "Funds for BHL-99 haven't arrived"}
    results = find_similar_tickets(target, TICKETS, top_k=10)
    assert all(t["status"] != "open" for t in results)


def test_excludes_self():
    target = TICKETS[0]
    results = find_similar_tickets(target, TICKETS, top_k=10)
    assert all(t["id"] != target["id"] for t in results)


def test_empty_pool_returns_empty_list():
    target = {"id": "T-99", "subject": "New", "body": "New ticket"}
    results = find_similar_tickets(target, [{"id": "T-99", "subject": "x", "body": "y", "status": "closed"}], top_k=3)
    assert results == []
