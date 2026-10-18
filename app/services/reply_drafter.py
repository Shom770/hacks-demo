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

    # Changed to ask for a JSON list of drafts instead of free text for each draft.
    # Set max_tokens to a reasonable value for a single draft.
    # Set temperature to 0 for deterministic output.
    prompt = DRAFT_PROMPT.format(
        master=build_master_prompt(),
        category=category,
        sentiment=mood,
        articles=articles,
        thread=thread,
    )
    drafts = anthropic_client.message(
        prompt,
        max_tokens=1000,  # Reduced max_tokens as we are now asking for a list of drafts
        response_format={"type": "json_object"}, # Request JSON output
        temperature=0 # Set temperature to 0 for deterministic output
    )
    # Assuming the model returns a JSON object with a key like "drafts" containing a list of strings
    # If the model returns a different structure, this parsing will need adjustment.
    drafts = drafts.get("drafts", []) 

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
        temperature=0.0,
        max_tokens=1024,
    )
