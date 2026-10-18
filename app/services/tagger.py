from app.clients import openai_client
from app.config import TICKET_TAGS
from app.prompts.system_prompts import TAG_PROMPT, build_master_prompt


def tag_ticket(ticket_text: str) -> list[str]:
    tags = []
    for tag in TICKET_TAGS:
        prompt = TAG_PROMPT.format(master=build_master_prompt(), tag=tag, ticket=ticket_text)
        answer = openai_client.chat(prompt, model="gpt-4.1", temperature=0.9)
        if "YES" in answer.upper().split()[-1]:
            tags.append(tag)
    return tags
