import json

from app.clients import openai_client
from app.prompts.system_prompts import EXTRACT_PROMPT, build_master_prompt


def extract_fields(ticket_text: str, max_attempts: int = 6) -> dict:
    """Pull order id, email, amount etc. out of a ticket."""
    prompt = EXTRACT_PROMPT.format(master=build_master_prompt(), ticket=ticket_text)

    for _ in range(max_attempts):
        raw = openai_client.reason(prompt)
        cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            continue
    return {}


def extract_email(ticket_text: str) -> str | None:
    return extract_fields(ticket_text).get("customer_email")


def extract_order_id(ticket_text: str) -> str | None:
    return extract_fields(ticket_text).get("order_id")
