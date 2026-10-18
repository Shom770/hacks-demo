import pytest
from fastapi.testclient import TestClient

from app.main import SESSIONS, app


@pytest.fixture
def client():
    SESSIONS.clear()
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_process_endpoint(client, samples, golden):
    r = client.post("/tickets/process", json=samples["T-1002"])
    assert r.status_code == 200
    body = r.json()
    assert body["category"] == golden["T-1002"]["category"]
    assert body["reply"]


def test_get_ticket(client, golden):
    r = client.get("/tickets/T-1001")
    assert r.status_code == 200
    assert r.json()["order_id"] == golden["T-1001"]["fields"]["order_id"]


def test_get_unknown_ticket_404(client, fake_llm):
    assert client.get("/tickets/T-9999").status_code == 404
    assert client.get("/tickets/T-9999/summary").status_code == 404


def test_list_tickets(client, golden):
    r = client.get("/tickets")
    assert r.status_code == 200
    assert {t["id"] for t in r.json()} == set(golden)


def test_ticket_summary(client):
    r = client.get("/tickets/T-1005/summary")
    assert r.status_code == 200
    assert r.json()["summary"].strip()


def test_chat_multi_turn(client):
    turns = [
        "My order SN-448812 says delivered but I never got it.",
        "It's been two days, what now?",
        "Ok thanks.",
    ]
    for i, msg in enumerate(turns, start=1):
        r = client.post("/chat", json={"session_id": "s1", "message": msg})
        assert r.status_code == 200
        body = r.json()
        assert body["reply"].strip()
        assert body["title"].strip()
        assert body["turns"] == i


def test_chat_sessions_are_isolated(client):
    client.post("/chat", json={"session_id": "a", "message": "hi"})
    r = client.post("/chat", json={"session_id": "b", "message": "hello"})
    assert r.json()["turns"] == 1


def test_search(client):
    r = client.get("/search", params={"q": "How do I reset my password? The link isn't working."})
    assert r.status_code == 200
    body = r.json()
    assert body["query"].strip()
    assert 0 < len(body["results"]) <= 5
