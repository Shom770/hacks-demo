"""Behavioural contracts for each AI-backed service.

These pin what each function returns, not how it gets there, so swapping the
model, provider, prompt, or merging calls should keep them green.
"""
import json

import pytest

from app.config import TICKET_TAGS
from app.services import classifier, extractor, kb_search, reply_drafter, sentiment, summarizer, tagger, translator
from tests.conftest import GOLDEN, SAMPLES

ENGLISH = [t for t in GOLDEN if t["language"] == "English"]
ids = lambda ts: [t["id"] for t in ts]  # noqa: E731


@pytest.mark.parametrize("t", GOLDEN, ids=ids(GOLDEN))
def test_classify_returns_expected_category(t):
    result = classifier.classify_ticket(t["body_en"])
    assert result in classifier.CATEGORIES
    assert result == t["category"]


def test_classify_many_keys_by_ticket_id():
    tickets = [SAMPLES[t["id"]] for t in ENGLISH[:3]]
    result = classifier.classify_many(tickets)
    assert set(result) == {t["id"] for t in tickets}


@pytest.mark.parametrize("t", GOLDEN, ids=ids(GOLDEN))
def test_sentiment_returns_expected_label(t):
    assert sentiment.get_sentiment(t["body_en"]) == t["sentiment"]


def test_sentiment_per_message_preserves_length_and_order():
    msgs = [GOLDEN[2]["body_en"], GOLDEN[4]["body_en"], GOLDEN[6]["body_en"]]
    result = sentiment.sentiment_per_message(msgs)
    assert result == [GOLDEN[2]["sentiment"], GOLDEN[4]["sentiment"], GOLDEN[6]["sentiment"]]


@pytest.mark.parametrize("t", GOLDEN, ids=ids(GOLDEN))
def test_tags_cover_required_and_stay_in_vocabulary(t):
    tags = tagger.tag_ticket(t["body_en"])
    assert isinstance(tags, list)
    assert set(tags) <= set(TICKET_TAGS)
    assert set(t["tags_required"]) <= set(tags)
    assert not set(tags) & set(t["tags_forbidden"])
    assert len(tags) == len(set(tags)), "duplicate tags"


@pytest.mark.parametrize("t", GOLDEN, ids=ids(GOLDEN))
def test_extract_fields(t):
    fields = extractor.extract_fields(t["body_en"])
    assert isinstance(fields, dict)
    for key in ("order_id", "customer_email"):
        assert fields.get(key) == t["fields"][key], key
    if t["fields"]["amount"] is not None:
        assert float(str(fields["amount"]).replace("$", "").replace(",", "")) == pytest.approx(t["fields"]["amount"])


@pytest.mark.parametrize("t", ENGLISH[:2], ids=ids(ENGLISH[:2]))
def test_extract_helpers_agree_with_extract_fields(t):
    assert extractor.extract_order_id(t["body_en"]) == t["fields"]["order_id"]
    assert extractor.extract_email(t["body_en"]) == t["fields"]["customer_email"]


@pytest.mark.parametrize("t", [t for t in GOLDEN if t["relevant_kb"]], ids=ids([t for t in GOLDEN if t["relevant_kb"]]))
def test_kb_search_surfaces_a_relevant_article(t):
    results = kb_search.search(t["body_en"], top_k=3)
    assert 0 < len(results) <= 3
    assert all({"id", "title", "body"} <= set(r) for r in results)
    assert {r["id"] for r in results} & set(t["relevant_kb"])


def test_kb_search_respects_top_k():
    assert len(kb_search.search(GOLDEN[0]["body_en"], top_k=1)) == 1


def test_kb_semantic_search_shape():
    results = kb_search.search_semantic("how do I reset my password", top_k=2)
    assert len(results) == 2
    assert all("id" in r for r in results)


def test_rewrite_query_returns_text():
    out = kb_search.rewrite_query(GOLDEN[5]["body_en"])
    assert isinstance(out, str) and out.strip()


@pytest.mark.parametrize("t", GOLDEN, ids=ids(GOLDEN))
def test_detect_language(t):
    body = SAMPLES[t["id"]]["body"]
    assert translator.detect_language(body).lower() == t["language"].lower()


def test_translate_spanish_ticket_to_english():
    t = next(t for t in GOLDEN if t["language"] == "Spanish")
    out = translator.translate(SAMPLES[t["id"]]["body"], "English")
    assert "SN-551203" in out
    assert "lamp" in out.lower()


def test_translate_reply_back_to_customer_language():
    t = next(t for t in GOLDEN if t["language"] == "Spanish")
    out = translator.translate(t["reference_reply"], "Spanish")
    assert out == t["reference_reply_translated"]


def test_summarize_thread_returns_summary():
    t = next(t for t in GOLDEN if t["id"] == "T-1005")
    thread = SAMPLES[t["id"]]["thread"] + [{"role": "customer", "content": t["body_en"]}]
    out = summarizer.summarize_with_check(thread)
    assert isinstance(out, str) and out.strip()


def test_summarize_empty_thread_does_not_crash():
    assert isinstance(summarizer.summarize_thread([]), str)


@pytest.mark.parametrize("t", GOLDEN, ids=ids(GOLDEN))
def test_draft_reply(t):
    ticket = {**SAMPLES[t["id"]], "body": t["body_en"]}
    reply = reply_drafter.draft_reply(ticket)
    assert isinstance(reply, str) and reply.strip()
    for word in ("WMS", "RMA code", "Tier 2"):
        assert word not in reply, f"internal jargon in reply: {word}"


# --- robustness to messy model output ----------------------------------------
# Real models wrap answers in prose and fences; changes must keep tolerating that.


def test_classifier_tolerates_reasoning_before_label(fake_llm):
    fake_llm.respond_with(lambda prompt, t: "The customer mentions a charge on their card.\n\nCategory: billing")
    assert classifier.classify_ticket(GOLDEN[1]["body_en"]) == "billing"


def test_extractor_tolerates_markdown_fenced_json(fake_llm):
    payload = {"order_id": "SN-448812", "customer_email": "maria.g@example.com",
               "product_name": None, "amount": None, "currency": None}
    fake_llm.respond_with(lambda prompt, t: "```json\n" + json.dumps(payload) + "\n```")
    assert extractor.extract_fields(GOLDEN[0]["body_en"])["order_id"] == "SN-448812"


def test_extractor_returns_dict_when_model_never_returns_json(fake_llm):
    fake_llm.respond_with(lambda prompt, t: "Sorry, I can't help with that.")
    assert extractor.extract_fields(GOLDEN[0]["body_en"]) == {}


def test_sentiment_falls_back_to_neutral_on_garbage(fake_llm):
    fake_llm.respond_with(lambda prompt, t: "¯\\_(ツ)_/¯")
    assert sentiment.get_sentiment(GOLDEN[0]["body_en"]) == "neutral"
