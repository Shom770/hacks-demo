from app.clients import anthropic_client
from app.prompts.system_prompts import CLASSIFIER_PROMPT, build_master_prompt

CATEGORIES = ["billing", "shipping", "account", "product", "other"]


def classify_ticket(ticket_text: str) -> str:
    prompt = (
        "Classify the following customer support ticket into exactly one of these categories: "
        "billing, shipping, account, product, other.\n"
        "Reply with ONLY the category name, nothing else.\n\n"
        f'Ticket:\n"""\n{ticket_text}\n"""'
    )
    raw = anthropic_client.message(prompt, max_tokens=10)

    last_line = raw.strip().splitlines()[-1].strip().lower()
    for c in CATEGORIES:
        if c in last_line:
            return c

    return "other"


def classify_many(tickets: list[dict]) -> dict:
    return {t["id"]: classify_ticket(t["body"]) for t in tickets}
