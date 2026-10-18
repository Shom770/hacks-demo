import json
import re

from app.clients import groq_client, openai_client
from app.config import load_text
from app.prompts.system_prompts import RELEVANCE_PROMPT, build_master_prompt


def load_articles() -> list[dict]:
    return json.loads(load_text("knowledge_base.json"))


def search(question: str, top_k: int = 3) -> list[dict]:
    """Rank every KB article by asking an LLM to score it."""
    scored = []
    for article in load_articles():
        prompt = (
            f"On a scale from 0 to 10, how relevant is the following knowledge base article to the "
            f"customer question?\n"
            f"Reply with only 'SCORE: <n>' where <n> is a number from 0 to 10.\n\n"
            f"Customer question: {question}\n\n"
            f"Article:\n{json.dumps(article, indent=2)}"
        )
        answer = openai_client.chat(prompt, model="gpt-4o", temperature=0, max_tokens=10)
        match = re.search(r"SCORE:\s*(\d+)", answer)
        score = int(match.group(1)) if match else 0
        scored.append((score, article))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [a for _, a in scored[:top_k]]


def search_semantic(question: str, top_k: int = 3) -> list[dict]:
    """Embedding-based variant (re-embeds the whole KB on every query)."""
    q_vec = openai_client.embed(question)
    scored = []
    for article in load_articles():
        a_vec = openai_client.embed(article["title"] + "\n" + article["body"])
        sim = sum(x * y for x, y in zip(q_vec, a_vec))
        scored.append((sim, article))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [a for _, a in scored[:top_k]]


def rewrite_query(question: str) -> str:
    return groq_client.fast_chat(
        f"Rewrite this customer question to be a better search query:\n{question}",
        system="You are SupportPilot, a world-class AI customer support assistant for ShopNest Inc. Rewrite the user's question to be a better search query. Be concise.",
        temperature=0,
    )
