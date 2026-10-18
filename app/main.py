import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.clients import anthropic_client, openai_client
from app.config import load_text
from app.pipeline import process_ticket
from app.prompts.system_prompts import build_master_prompt
from app.services import kb_search, summarizer

app = FastAPI(title="SupportPilot")

# In-memory chat sessions: session_id -> list of messages
SESSIONS: dict[str, list[dict]] = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str


class TicketRequest(BaseModel):
    id: str
    body: str
    thread: list[dict] = []


def _tickets() -> list[dict]:
    return json.loads(load_text("sample_tickets.json"))


@app.get("/health")
def health():
    reply = openai_client.chat("Reply with the word OK.", model="gpt-4.1", system=build_master_prompt())
    return {"status": "ok" if "OK" in reply.upper() else "degraded", "llm": reply}


@app.post("/tickets/process")
def process(req: TicketRequest):
    return process_ticket(req.model_dump())


@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: str):
    for t in _tickets():
        if t["id"] == ticket_id:
            return process_ticket(t)
    raise HTTPException(404, "ticket not found")


@app.get("/tickets")
def list_tickets():
    # Return processed tickets so the dashboard has everything it needs
    return [process_ticket(t) for t in _tickets()]


@app.get("/tickets/{ticket_id}/summary")
def ticket_summary(ticket_id: str):
    for t in _tickets():
        if t["id"] == ticket_id:
            thread = t.get("thread", []) + [{"role": "customer", "content": t["body"]}]
            return {"summary": summarizer.summarize_with_check(thread)}
    raise HTTPException(404, "ticket not found")


@app.post("/chat")
def chat(req: ChatRequest):
    history = SESSIONS.setdefault(req.session_id, [])

    articles = kb_search.search(req.message)
    prompt = f"Relevant articles: {articles}\n\nCustomer: {req.message}"

    reply = anthropic_client.message(
        prompt,
        system=build_master_prompt(),
        history=history,
        max_tokens=32000,
    )

    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": reply})

    # Keep a running title for the session
    title = openai_client.reason(
        "Give a 3-word title for this conversation:\n" + json.dumps(history)
    )
    return {"reply": reply, "title": title, "turns": len(history) // 2}


@app.get("/search")
def search(q: str):
    rewritten = kb_search.rewrite_query(q)
    return {"query": rewritten, "results": kb_search.search(rewritten, top_k=5)}
