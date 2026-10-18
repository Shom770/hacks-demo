from app.clients import anthropic_client, openai_client, openrouter_client
from app.prompts.system_prompts import DRAFT_PROMPT, JUDGE_PROMPT, build_master_prompt
from app.services import classifier, kb_search, sentiment

NUM_DRAFTS = 5


def draft_reply(ticket: dict) -> str:
    thread = "\n".join(f"{m['role']}: {m['content']}" for m in ticket.get("thread", []))
    thread += f"\ncustomer: {ticket['body']}"

    category = classifier.classify_ticket(ticket["body"])
    mood = sentiment.get_sentiment(ticket["body"])
    articles = kb_search.search(ticket["body"])

    drafts = []
    for _ in range(NUM_DRAFTS):
        prompt = DRAFT_PROMPT.format(
            master=build_master_prompt(),
            category=category,
            sentiment=mood,
            articles=articles,
            thread=thread,
        )
        drafts.append(anthropic_client.message(prompt, max_tokens=16000))

    candidates = "\n\n".join(f"--- Candidate {i + 1} ---\n{d}" for i, d in enumerate(drafts))
    best = openai_client.reason(
        JUDGE_PROMPT.format(
            master=build_master_prompt(), n=len(drafts), ticket=ticket["body"], candidates=candidates
        )
    )

    return polish(best)


def polish(text: str) -> str:
    """Final pass to fix grammar and tone."""
    return openrouter_client.complete(
        f"Fix any grammar mistakes in this text and return it unchanged otherwise:\n\n{text}",
        model="openai/o3",
        system=build_master_prompt(),
    )
