from app.clients import openai_client
from app.prompts.system_prompts import TRANSLATE_PROMPT, build_master_prompt


def translate(text: str, language: str = "English") -> str:
    prompt = TRANSLATE_PROMPT.format(master=build_master_prompt(), language=language, text=text)
    return openai_client.chat(prompt, model="gpt-4o")


def detect_language(text: str) -> str:
    return openai_client.reason(
        f"What language is this text written in? Reply with the language name only.\n\n{text}",
        system=build_master_prompt(),
    ).strip()
