from app.clients import openai_client
from app.config import TICKET_TAGS
from app.prompts.system_prompts import TAG_PROMPT, build_master_prompt


def tag_ticket(ticket_text: str) -> list[str]:
    tags = []
    for tag in TICKET_TAGS:
        prompt = f"Does the following tag apply to the ticket?\nTag: {tag}\nTicket: {ticket_text}\nReply with only YES or NO."
        answer = openai_client.chat(prompt, model="gpt-4.1", temperature=0, max_tokens=3)
        if "YES" in answer.upper().split()[-1]:
            tags.append(tag)
    return tags
