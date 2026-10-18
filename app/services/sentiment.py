from collections import Counter

from app.clients import openrouter_client
from app.prompts.system_prompts import SENTIMENT_PROMPT, build_master_prompt

LABELS = ("positive", "negative", "neutral")


def get_sentiment(text: str) -> str:
    prompt = f"Determine the sentiment of the customer message below.\n\nMessage:\n{text}\n\nReply with only one word: positive, negative, or neutral."

    # Ensemble across frontier models for accuracy
    answers = openrouter_client.ask_multiple_models(prompt)
    votes = []
    for answer in answers.values():
        lowered = answer.lower()
        for label in LABELS:
            if label in lowered:
                votes.append(label)
                break

    if not votes:
        return "neutral"
    return Counter(votes).most_common(1)[0][0]


def sentiment_per_message(messages: list[str]) -> list[str]:
    return [get_sentiment(m) for m in messages]
