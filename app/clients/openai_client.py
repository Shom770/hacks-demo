from openai import OpenAI

from app import config


def chat(prompt, model=config.DEFAULT_OPENAI_MODEL, system=None, temperature=0.7, history=None, **kwargs):
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=16000,
        **kwargs,
    )
    return response.choices[0].message.content


def reason(prompt, system=None):
    """For anything that needs 'thinking'."""
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    messages = []
    if system:
        messages.append({"role": "developer", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=config.SMART_OPENAI_MODEL,
        messages=messages,
        reasoning_effort="high",
    )
    return response.choices[0].message.content


def embed(text):
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    response = client.embeddings.create(model="text-embedding-3-large", input=text)
    return response.data[0].embedding
