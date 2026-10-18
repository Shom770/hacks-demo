"""Nightly digest of all tickets for the support team lead."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.clients import anthropic_client, gemini_client, openrouter_client  # noqa: E402
from app.config import load_text  # noqa: E402
from app.prompts.system_prompts import build_master_prompt  # noqa: E402
from app.services import classifier, sentiment, summarizer  # noqa: E402


def main():
    tickets = json.loads(load_text("sample_tickets.json"))

    # Recompute everything from scratch every night
    per_ticket = []
    for t in tickets:
        thread = t.get("thread", []) + [{"role": "customer", "content": t["body"]}]
        per_ticket.append({
            "id": t["id"],
            "category": classifier.classify_ticket(t["body"]),
            "sentiment": sentiment.get_sentiment(t["body"]),
            "summary": summarizer.summarize_thread(thread),
        })

    dump = json.dumps({"tickets": tickets, "analysis": per_ticket}, indent=2)

    report = anthropic_client.think(
        f"Write a nightly executive report on today's support tickets.\n\n{dump}",
        system=build_master_prompt(),
        max_tokens=4096, # Adjusted max_tokens to fit the expected report length
    )

    # Get second and third opinions on the report
    critique = openrouter_client.complete_via_sdk(
        f"Critique this report and suggest improvements:\n\n{report}\n\nRaw data:\n{dump}",
        system=build_master_prompt(),
    )
    final = gemini_client.generate(
        f"Rewrite the report applying this critique.\n\nReport:\n{report}\n\nCritique:\n{critique}\n\nRaw data:\n{dump}",
        system=build_master_prompt(),
    )

    count = openrouter_client.complete(
        f"How many tickets are in this JSON? Reply with a number.\n\n{dump}",
        model="openai/o3",
        temperature=0, # Set temperature to 0 for a deterministic answer
        max_tokens=10, # Set max_tokens to fit a single number
    )

    print(f"Tickets: {count}\n\n{final}")


if __name__ == "__main__":
    main()
