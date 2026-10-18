import pytest

from app.config import TICKET_TAGS
from app.pipeline import process_ticket
from app.services.classifier import CATEGORIES
from tests.conftest import GOLDEN

EXPECTED_KEYS = {
    "id", "language", "category", "sentiment", "thread_sentiments", "tags",
    "order_id", "email", "fields", "summary", "reply",
}


@pytest.mark.parametrize("t", GOLDEN, ids=[t["id"] for t in GOLDEN])
def test_process_ticket_output_contract(t, samples):
    ticket = samples[t["id"]]
    out = process_ticket(ticket)

    assert set(out) >= EXPECTED_KEYS
    assert out["id"] == t["id"]
    assert out["language"].lower() == t["language"].lower()
    assert out["category"] in CATEGORIES and out["category"] == t["category"]
    assert out["sentiment"] == t["sentiment"]
    assert set(out["tags"]) <= set(TICKET_TAGS)
    assert set(t["tags_required"]) <= set(out["tags"])
    assert out["order_id"] == t["fields"]["order_id"]
    assert out["email"] == t["fields"]["customer_email"]
    assert isinstance(out["fields"], dict)
    assert isinstance(out["summary"], str) and out["summary"].strip()
    assert isinstance(out["reply"], str) and out["reply"].strip()
    assert len(out["thread_sentiments"]) == len(ticket.get("thread", [])) + 1


def test_reply_is_in_customer_language(samples, golden):
    t = golden["T-1004"]
    out = process_ticket(samples["T-1004"])
    assert out["reply"] == t["reference_reply_translated"]


def test_process_ticket_does_not_mutate_input(samples):
    ticket = samples["T-1004"]
    snapshot = {**ticket, "thread": list(ticket["thread"])}
    process_ticket(ticket)
    assert ticket == snapshot
