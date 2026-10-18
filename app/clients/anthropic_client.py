import anthropic

from app import config


def message(prompt, system="", model=config.DEFAULT_CLAUDE_MODEL, max_tokens=8192, history=None):
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY, timeout=1800)

    messages = list(history or [])
    messages.append({"role": "user", "content": prompt})

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    )
    return "".join(block.text for block in response.content if block.type == "text")


def think(prompt, system=""):
    """Extended thinking for harder problems."""
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY, timeout=1800)
    response = client.messages.create(
        model=config.DEFAULT_CLAUDE_MODEL,
        max_tokens=64000,
        thinking={"type": "enabled", "budget_tokens": 32000},
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")
