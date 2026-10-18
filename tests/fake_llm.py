"""Transport-level fake for every LLM provider the app talks to.

Requests are intercepted below the SDKs (httpx for openai/anthropic/groq,
requests for openrouter/google-genai), so tests keep working when a change
swaps the model, the provider, the SDK, or rewrites the prompt.

Answers come from an oracle that finds the golden ticket in the prompt and
works out which task is being asked for. If a change drops the ticket text from
a prompt, the oracle can't find it and the test fails, which is intentional.
"""
import base64
import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx
import requests

from app.config import TICKET_TAGS, load_text

CATEGORIES = ["billing", "shipping", "account", "product", "other"]
KNOWN_HOSTS = {
    "api.openai.com": "openai",
    "openrouter.ai": "openrouter",
    "api.groq.com": "groq",
    "api.anthropic.com": "anthropic",
    "generativelanguage.googleapis.com": "gemini",
}
TEXT_KEYS = {"content", "text", "system", "input", "prompt", "instructions", "system_instruction", "systemInstruction"}


def est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _collect_text(node, out, under_text_key=False):
    if isinstance(node, str):
        if under_text_key:
            out.append(node)
    elif isinstance(node, dict):
        for k, v in node.items():
            _collect_text(v, out, under_text_key or k in TEXT_KEYS)
    elif isinstance(node, list):
        for v in node:
            _collect_text(v, out, under_text_key)


def fake_embedding(text: str, dims: int = 256) -> list[float]:
    vec = [0.0] * dims
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % dims] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@dataclass
class Call:
    provider: str
    endpoint: str
    model: str
    input_tokens: int
    output_tokens: int
    max_tokens: int | None
    prompt: str = field(repr=False)


class FakeLLM:
    def __init__(self, golden: list[dict]):
        self.golden = golden
        self.calls: list[Call] = []
        self.unexpected: list[str] = []
        self._override = None
        self._boilerplate = self._load_boilerplate()

    # ----- public helpers -------------------------------------------------

    def respond_with(self, fn):
        """fn(prompt, ticket) -> str | None. Return None to fall back to the oracle."""
        self._override = fn

    def reset(self):
        self.calls.clear()

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    def usage(self) -> dict:
        by_model: dict[str, int] = {}
        for c in self.calls:
            by_model[c.model] = by_model.get(c.model, 0) + 1
        return {
            "calls": len(self.calls),
            "input_tokens": self.total_input_tokens,
            "output_tokens": sum(c.output_tokens for c in self.calls),
            "calls_by_model": dict(sorted(by_model.items())),
        }

    def install(self, monkeypatch):
        fake = self

        def httpx_handle(transport, request: httpx.Request):
            request.read()
            status, body = fake._dispatch(str(request.url), request.content)
            return httpx.Response(status, json=body, request=request)

        async def httpx_handle_async(transport, request: httpx.Request):
            await request.aread()
            status, body = fake._dispatch(str(request.url), request.content)
            return httpx.Response(status, json=body, request=request)

        def requests_send(adapter, request, **kwargs):
            raw = request.body or b""
            status, body = fake._dispatch(request.url, raw.encode() if isinstance(raw, str) else raw)
            resp = requests.Response()
            resp.status_code = status
            resp._content = json.dumps(body).encode()
            resp.headers["Content-Type"] = "application/json"
            resp.encoding = "utf-8"
            resp.url = request.url
            resp.request = request
            return resp

        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", httpx_handle)
        monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", httpx_handle_async)
        monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", requests_send)

    # ----- routing --------------------------------------------------------

    def _dispatch(self, url: str, raw: bytes):
        parsed = urlparse(url)
        host, path = parsed.netloc, parsed.path
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            payload = {}

        # Route on the API path, not the host, so provider base URLs pointed at a
        # gateway/proxy (OPENAI_BASE_URL, ANTHROPIC_BASE_URL, ...) still hit the fake.
        provider = KNOWN_HOSTS.get(host, host)
        if path.endswith("/chat/completions"):
            return 200, self._openai_chat(provider, payload)
        if path.endswith("/responses"):
            return 200, self._openai_responses(provider, payload)
        if path.endswith("/embeddings"):
            return 200, self._openai_embeddings(provider, payload)
        if path.endswith("/messages/count_tokens"):
            return 200, {"input_tokens": est_tokens(self._prompt_text(payload))}
        if path.endswith("/messages"):
            return 200, self._anthropic(payload)
        if ":generateContent" in path:
            model = path.split("/models/")[-1].split(":")[0]
            return 200, self._gemini(model, payload)

        self.unexpected.append(f"{host}{path}")
        return 404, {"error": {"message": f"fake_llm has no route for {host}{path}", "type": "not_found"}}

    # ----- provider response shapes ----------------------------------------

    def _record(self, provider, endpoint, model, prompt, answer, max_tokens):
        self.calls.append(Call(provider, endpoint, model, est_tokens(prompt), est_tokens(answer), max_tokens, prompt))

    def _openai_chat(self, provider, payload):
        prompt = self._prompt_text(payload)
        wants_json = "response_format" in payload
        n = payload.get("n") or 1
        answer = self.answer(prompt, wants_json)
        self._record(provider, "chat.completions", payload.get("model", "?"), prompt, answer * n,
                     payload.get("max_tokens") or payload.get("max_completion_tokens"))
        return {
            "id": "chatcmpl-fake", "object": "chat.completion", "created": 0, "model": payload.get("model"),
            "choices": [
                {"index": i, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}
                for i in range(n)
            ],
            "usage": {"prompt_tokens": est_tokens(prompt), "completion_tokens": est_tokens(answer),
                      "total_tokens": est_tokens(prompt) + est_tokens(answer)},
        }

    def _openai_responses(self, provider, payload):
        prompt = self._prompt_text(payload)
        answer = self.answer(prompt, "text" in payload and "format" in (payload.get("text") or {}))
        self._record(provider, "responses", payload.get("model", "?"), prompt, answer, payload.get("max_output_tokens"))
        return {
            "id": "resp_fake", "object": "response", "created_at": 0, "model": payload.get("model"),
            "status": "completed", "error": None, "incomplete_details": None, "instructions": None,
            "metadata": {}, "parallel_tool_calls": True, "temperature": 1.0, "top_p": 1.0,
            "tool_choice": "auto", "tools": [],
            "output": [{
                "type": "message", "id": "msg_fake", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": answer, "annotations": []}],
            }],
            "usage": {"input_tokens": est_tokens(prompt), "output_tokens": est_tokens(answer),
                      "total_tokens": est_tokens(prompt) + est_tokens(answer),
                      "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 0}},
        }

    def _openai_embeddings(self, provider, payload):
        inputs = payload.get("input")
        inputs = [inputs] if isinstance(inputs, str) else list(inputs or [])
        b64 = payload.get("encoding_format") == "base64"
        data = []
        for i, text in enumerate(inputs):
            vec = fake_embedding(str(text))
            emb = base64.b64encode(struct.pack(f"<{len(vec)}f", *vec)).decode() if b64 else vec
            data.append({"object": "embedding", "index": i, "embedding": emb})
        joined = "\n".join(map(str, inputs))
        self._record(provider, "embeddings", payload.get("model", "?"), joined, "", None)
        return {"object": "list", "data": data, "model": payload.get("model"),
                "usage": {"prompt_tokens": est_tokens(joined), "total_tokens": est_tokens(joined)}}

    def _anthropic(self, payload):
        prompt = self._prompt_text(payload)
        answer = self.answer(prompt, "output_format" in payload)
        self._record("anthropic", "messages", payload.get("model", "?"), prompt, answer, payload.get("max_tokens"))
        return {
            "id": "msg_fake", "type": "message", "role": "assistant", "model": payload.get("model"),
            "content": [{"type": "text", "text": answer}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": est_tokens(prompt), "output_tokens": est_tokens(answer)},
        }

    def _gemini(self, model, payload):
        prompt = self._prompt_text(payload)
        cfg = payload.get("generationConfig") or {}
        answer = self.answer(prompt, cfg.get("responseMimeType") == "application/json")
        self._record("gemini", "generateContent", model, prompt, answer, cfg.get("maxOutputTokens"))
        return {
            "candidates": [{"content": {"parts": [{"text": answer}], "role": "model"}, "finishReason": "STOP", "index": 0}],
            "usageMetadata": {"promptTokenCount": est_tokens(prompt), "candidatesTokenCount": est_tokens(answer),
                              "totalTokenCount": est_tokens(prompt) + est_tokens(answer)},
        }

    # ----- oracle ---------------------------------------------------------

    def _load_boilerplate(self) -> list[str]:
        """Static context (system prompt, policy, KB) stripped before reading the task."""
        chunks = []
        try:
            from app.prompts.system_prompts import build_master_prompt
            chunks.append(build_master_prompt())
        except Exception:
            pass
        for name in ("company_policy.md", "knowledge_base.json"):
            try:
                chunks.append(load_text(name))
            except Exception:
                pass
        return sorted(chunks, key=len, reverse=True)

    def _prompt_text(self, payload) -> str:
        out: list[str] = []
        _collect_text(payload, out)
        return "\n".join(out)

    def _strip(self, prompt: str) -> str:
        for chunk in self._boilerplate:
            prompt = prompt.replace(chunk, " ")
        return prompt

    def find_ticket(self, text: str):
        low = text.lower()
        for t in self.golden:
            if t.get("reference_reply_translated") and t["reference_reply_translated"].lower()[:60] in low:
                return t, "reply"
            if t["reference_reply"].lower()[:60] in low:
                return t, "reply"
            if any(m in low for m in t["match"]):
                return t, "body"
        return None, None

    def _json_answer(self, t):
        if t is None:
            return json.dumps({"category": "other", "sentiment": "neutral", "tags": [], "language": "English",
                               "order_id": None, "customer_email": None, "product_name": None,
                               "amount": None, "currency": None})
        return json.dumps({
            "category": t["category"], "sentiment": t["sentiment"], "tags": t["tags_required"],
            "language": t["language"], "product_name": None, **t["fields"],
            "summary": t["summary"], "relevant_kb": t["relevant_kb"],
        })

    def answer(self, prompt: str, wants_json: bool = False) -> str:
        task = self._strip(prompt)
        low = task.lower()
        t, kind = self.find_ticket(task)

        if self._override:
            out = self._override(task, t)
            if out is not None:
                return out

        if "word ok" in low:
            return "OK"
        if "how many tickets" in low:
            return str(len(self.golden))
        if wants_json or "json" in low:
            return self._json_answer(t)
        if "relevan" in low and re.search(r"\bscore\b|0 to 10|\brate\b", low):
            hit = t and any(kb in low for kb in t["relevant_kb"])
            return "Relevant." + ("\nSCORE: 9" if hit else "\nSCORE: 1")
        if "summary accurate" in low or "is this summary" in low:
            return "YES"
        if "summar" in low:
            return t["summary"] if t else "Customer conversation."
        if "what language" in low or "detect the language" in low:
            return t["language"] if t else "English"
        if "translate" in low:
            target = re.search(r"\b(?:into|to)\s+([a-z]+)", low)
            target = target.group(1) if target else "english"
            if not t:
                return "Translated text."
            if kind == "reply":
                if target != "english" and t.get("reference_reply_translated"):
                    return t["reference_reply_translated"]
                return t["reference_reply"]
            return t["body_en"]
        if "candidate" in low or "grammar" in low or "reply to the customer" in low or "draft" in low:
            if t and t.get("reference_reply_translated") and t["language"].lower() in low:
                return t["reference_reply_translated"]
            return t["reference_reply"] if t else "Thanks for reaching out, we're on it."
        if "sentiment" in low:
            return t["sentiment"] if t else "neutral"
        single_tag = re.search(r'tag\s+"([a-z_]+)"', low)
        if single_tag:
            return "YES" if t and single_tag.group(1) in t["tags_required"] else "NO"
        if sum(tag in low for tag in TICKET_TAGS) > 3 and "tag" in low:
            return ", ".join(t["tags_required"]) if t else ""
        if "categor" in low or "classif" in low:
            return t["category"] if t else "other"
        if "search query" in low:
            return t["body_en"] if t else task.strip()[-200:]
        if "title" in low:
            return "Support conversation"
        if "report" in low or "critique" in low:
            return "Daily report: tickets reviewed, no anomalies."
        return t["reference_reply"] if t else "Thanks for reaching out, we're on it."
