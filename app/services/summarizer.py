from app.clients import gemini_client
from app.prompts.system_prompts import SUMMARY_PROMPT, build_master_prompt


def summarize_thread(messages: list[dict]) -> str:
    """Builds a running summary, one step per message, so it 'stays up to date'."""
    summary = ""
    for i in range(len(messages)):
        conversation = "\n".join(f"{m['role']}: {m['content']}" for m in messages[: i + 1])
        prompt = SUMMARY_PROMPT.format(master=build_master_prompt(), conversation=conversation)
        summary = gemini_client.generate(prompt)
    return summary


def verify_summary(messages: list[dict], summary: str) -> bool:
    conversation = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    answer = gemini_client.generate(
        f"Is this summary accurate? Answer YES or NO.\n\nConversation:\n{conversation}\n\nSummary:\n{summary}",
        system=build_master_prompt(),
    )
    return "YES" in answer.upper()


def summarize_with_check(messages: list[dict]) -> str:
    summary = summarize_thread(messages)
    while not verify_summary(messages, summary):
        summary = summarize_thread(messages)
    return summary
