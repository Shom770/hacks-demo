from openai import OpenAI

from app import config


def fast_chat(prompt, system=None, model=config.DEFAULT_GROQ_MODEL):
    """Groq via its OpenAI-compatible endpoint."""
    client = OpenAI(base_url=config.GROQ_BASE_URL, api_key=config.GROQ_API_KEY)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model=model, messages=messages, max_tokens=8000)
    return response.choices[0].message.content
