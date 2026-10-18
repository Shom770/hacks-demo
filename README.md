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
