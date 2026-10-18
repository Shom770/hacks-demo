# SupportPilot

An AI-powered customer support assistant for ShopNest. It reads incoming tickets, figures out what they're about, pulls the relevant help articles, and drafts a reply.

## Features
- Ticket classification, sentiment analysis, and auto-tagging
- Field extraction (order ID, email, amount)
- Knowledge base search
- Thread summarization
- Multi-draft reply generation with LLM-as-judge selection
- Automatic translation
- Live chat endpoint
- Nightly executive report

## Providers
| Provider   | Access method                         | File |
|------------|---------------------------------------|------|
| OpenAI     | `openai` SDK                          | `app/clients/openai_client.py` |
| OpenRouter | raw `requests` + `openai` SDK         | `app/clients/openrouter_client.py` |
| Anthropic  | `anthropic` SDK                       | `app/clients/anthropic_client.py` |
| Google     | `google-genai` SDK                    | `app/clients/gemini_client.py` |
| Groq       | `openai` SDK (OpenAI-compatible URL)  | `app/clients/groq_client.py` |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in keys
```

## Run
```bash
uvicorn app.main:app --reload       # API
python scripts/run_demo.py          # process all sample tickets
python scripts/nightly_report.py    # nightly digest
```

## Layout
```
app/
  clients/     one wrapper per LLM provider
  services/    classifier, sentiment, tagger, extractor, kb_search, translator, summarizer, reply_drafter
  prompts/     system + task prompts
  pipeline.py  end-to-end ticket processing
  main.py      FastAPI app
data/          policy handbook, knowledge base, sample tickets
scripts/       batch demo + nightly report
```

## Tests
```bash
pip install -r requirements-dev.txt
pytest                                      # offline: contracts, pipeline, API, LLM cost budget
RUN_LIVE_EVALS=1 pytest -m live             # real providers: accuracy evals on the golden set
UPDATE_LLM_BASELINE=1 pytest tests/test_llm_budget.py   # re-record cost baseline
```
- Offline tests fake every provider at the HTTP layer (`tests/fake_llm.py`), so they survive model/provider/SDK/prompt changes. Calls to an unknown endpoint fail the test.
- Expected answers live in `tests/golden/tickets.json`.
- `tests/test_llm_budget.py` fails if a change increases LLM calls or input tokens vs `tests/baselines/llm_usage.json`, and writes a before/after diff to `output/llm_usage.json`.
