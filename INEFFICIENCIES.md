# Answer key: planted inefficiencies

> Delete this file before demoing if you want a detector to find these on its own.

One single-message ticket through `process_ticket` makes **~66 LLM calls**. T-1005 (5-message thread) makes **~90**. A sensible version needs 1–2.

## Bloated prompts
- `prompts/system_prompts.py` `MASTER_SYSTEM_PROMPT`: repeated "IMPORTANT" lines, identity restated three times, tipping/"deep breath" filler, "think step by step" for one-word answers.
- The **entire policy handbook + entire KB** is put into *every* prompt (classify, sentiment, tags, translate, grammar fix, health check, query rewrite...).
- `build_master_prompt()` re-reads both files from disk on every call.
- `CLASSIFIER_PROMPT` asks for a 3-paragraph analysis, then "ONLY the category" (contradictory, wastes output tokens).
- `nightly_report.py` sends the full raw JSON dump into 4 separate calls, including one just to **count tickets**.
- `/chat` stores the KB-stuffed prompt in history, so every turn re-sends all previous article dumps.

## Wrong model for the job
- Classification into 5 labels → `claude-opus-5-5` with 8k max_tokens, falls back to **extended thinking** (32k budget).
- Sentiment (3 labels) → **5-model frontier ensemble** via OpenRouter (o3, Opus, Gemini Pro, Grok 4, Llama 405B).
- Language detection → `o3` with `reasoning_effort="high"`.
- Field extraction (order ID / email) → `o3`; regex would do it.
- Grammar polish → `openai/o3`.
- Health check → calls `gpt-4.1` with the full master prompt.
- 3-word chat title → `o3`, sending the whole JSON history every turn.
- `text-embedding-3-large` where `-small` would do.

## Redundant / duplicate calls
- `pipeline.py` classifies and gets sentiment, then `reply_drafter.draft_reply` does both **again**.
- `extract_order_id`, `extract_email`, `extract_fields` each run a full extraction (3× the same call).
- Translation always runs, even for English (`TRANSLATE_PROMPT` says "translate it anyway"); language detected via LLM.
- `/tickets` (list endpoint) runs the full pipeline on every ticket on every GET.
- `nightly_report.py` recomputes everything from scratch, then gets a critique + rewrite with the raw data re-sent.

## N+1 / looping patterns
- `tagger.tag_ticket`: **20 calls** per ticket (one yes/no per tag) instead of one multi-label call.
- `kb_search.search`: one LLM relevance call **per KB article** (14) instead of embeddings.
- `kb_search.search_semantic`: re-embeds the whole KB on every query, one article per request (no batching, no cache).
- `summarizer.summarize_thread`: re-summarizes the prefix for every message → O(n²) tokens; only the last result is used.
- `summarize_with_check`: unbounded `while` loop re-running the quadratic summary until a verifier says YES.
- `sentiment_per_message`: 5-model ensemble × every message in the thread.
- `reply_drafter`: 5 drafts in a Python loop (instead of `n=5`), each with the full prompt, then an o3 judge that echoes the full winning text back.

## Parameter / config waste
- `max_tokens` set very high everywhere (16k OpenAI, 32k OpenRouter, 64k Gemini, 32k chat).
- `temperature=0.9`/`1.0` on classification-type tasks → inconsistent outputs → more retries.
- No `response_format` / structured outputs for JSON; up to **6 retries** re-sending the full prompt.
- No prompt caching on Anthropic despite a huge static system prompt.
- No result caching / memoization anywhere.
- A new SDK client is constructed on every call (no connection reuse); `requests.post` without a `Session` or timeout.
- OpenRouter retry loop: 5 attempts, 0.1s fixed sleep, retries on non-retryable errors (e.g. 401/400).
- Everything sequential; nothing parallelized or batched (Batch API would suit the nightly job).
