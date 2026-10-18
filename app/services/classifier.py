from app.clients import anthropic_client
from app.prompts.system_prompts import CLASSIFIER_PROMPT, build_master_prompt

CATEGORIES = ["billing", "shipping", "account", "product", "other"]


def classify_ticket(ticket_text: str) -> str:
    prompt = CLASSIFIER_PROMPT.format(master=build_master_prompt(), ticket=ticket_text)
    raw = anthropic_client.message(prompt, max_tokens=8192)

    last_line = raw.strip().splitlines()[-1].strip().lower()
    for c in CATEGORIES:
        if c in last_line:
            return c

    # Didn't get a clean answer, ask again with extended thinking
    raw = anthropic_client.think(prompt)
    for c in CATEGORIES:
        if c in raw.lower():
            return c
    return "other"


def classify_many(tickets: list[dict]) -> dict:
    return {t["id"]: classify_ticket(t["body"]) for t in tickets}
