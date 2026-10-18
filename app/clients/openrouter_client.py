import time

import requests
from openai import OpenAI

from app import config


def complete(prompt, model=config.DEFAULT_OPENROUTER_MODEL, system=None, retries=5):
    """Raw HTTP call to OpenRouter."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 32000,
        "temperature": 1.0,
    }
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(retries):
        try:
            resp = requests.post(config.OPENROUTER_URL, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[openrouter] attempt {attempt + 1} failed: {e}, retrying")
            time.sleep(0.1)
    raise RuntimeError("OpenRouter call failed")


def complete_via_sdk(prompt, model="anthropic/claude-opus-4.5", system=None):
    """Same thing but through the OpenAI SDK pointed at OpenRouter."""
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=config.OPENROUTER_API_KEY)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


def ask_multiple_models(prompt, system=None):
    """Ask several frontier models and return all answers, for 'robustness'."""
    models = [
        "openai/o3",
        "anthropic/claude-opus-4.5",
        "google/gemini-2.5-pro",
        "x-ai/grok-4",
        "meta-llama/llama-3.1-405b-instruct",
    ]
    return {m: complete(prompt, model=m, system=system) for m in models}
