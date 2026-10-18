"""Quality evals against real providers.

Skipped by default. Run with:
    RUN_LIVE_EVALS=1 pytest -m live

Thresholds are aggregate (accuracy over the golden set) rather than per-ticket,
since model output varies run to run.
"""
import re

import pytest

from app.config import TICKET_TAGS
from app.pipeline import process_ticket
from app.services import classifier, extractor, kb_search, reply_drafter, sentiment, tagger, translator
from tests.conftest import GOLDEN, SAMPLES

pytestmark = pytest.mark.live

MIN_CLASSIFY_ACC = 0.85
MIN_SENTIMENT_ACC = 0.85
MIN_TAG_RECALL = 0.8
MIN_KB_HIT_RATE = 0.8
MAX_REPLY_WORDS = 200
JARGON = ("WMS", "RMA code", "Tier 2")
SPANISH_MARKERS = {"el", "la", "de", "que", "su", "por", "para", "pedido", "lamentamos", "gracias"}


def _accuracy(pairs):
    return sum(ok for ok in pairs) / len(pairs)


def test_classification_accuracy():
    hits = [classifier.classify_ticket(t["body_en"]) in t["category_allowed"] for t in GOLDEN]
    assert _accuracy(hits) >= MIN_CLASSIFY_ACC


def test_sentiment_accuracy():
    hits = [sentiment.get_sentiment(t["body_en"]) in t["sentiment_allowed"] for t in GOLDEN]
    assert _accuracy(hits) >= MIN_SENTIMENT_ACC


def test_tag_recall_and_no_forbidden_tags():
    required = found = 0
    for t in GOLDEN:
        tags = set(tagger.tag_ticket(t["body_en"]))
        assert tags <= set(TICKET_TAGS), f"{t['id']}: tags outside vocabulary: {tags - set(TICKET_TAGS)}"
        assert not tags & set(t["tags_forbidden"]), f"{t['id']}: forbidden tags {tags & set(t['tags_forbidden'])}"
        required += len(t["tags_required"])
        found += len(tags & set(t["tags_required"]))
    assert found / required >= MIN_TAG_RECALL


@pytest.mark.parametrize("t", GOLDEN, ids=[t["id"] for t in GOLDEN])
def test_extraction_exact(t):
    fields = extractor.extract_fields(t["body_en"])
    assert fields.get("order_id") == t["fields"]["order_id"]
    assert fields.get("customer_email") == t["fields"]["customer_email"]
    if t["fields"]["amount"] is not None:
        amount = float(re.sub(r"[^\d.]", "", str(fields.get("amount"))))
        assert amount == pytest.approx(t["fields"]["amount"])


def test_language_detection():
    for t in GOLDEN:
        assert t["language"].lower() in translator.detect_language(SAMPLES[t["id"]]["body"]).lower()


def test_kb_hit_rate():
    cases = [t for t in GOLDEN if t["relevant_kb"]]
    hits = [bool({r["id"] for r in kb_search.search(t["body_en"])} & set(t["relevant_kb"])) for t in cases]
    assert _accuracy(hits) >= MIN_KB_HIT_RATE


@pytest.mark.parametrize("tid", ["T-1001", "T-1005", "T-1008"])
def test_reply_quality(tid):
    t = next(g for g in GOLDEN if g["id"] == tid)
    reply = reply_drafter.draft_reply({**SAMPLES[tid], "body": t["body_en"]})
    assert reply.strip()
    assert len(reply.split()) <= MAX_REPLY_WORDS
    assert not [j for j in JARGON if j in reply]
    if t["fields"]["order_id"]:
        assert t["fields"]["order_id"] in reply


def test_spanish_ticket_gets_spanish_reply():
    out = process_ticket(SAMPLES["T-1004"])
    words = set(re.findall(r"[a-záéíóúñ]+", out["reply"].lower()))
    assert len(words & SPANISH_MARKERS) >= 3, out["reply"]
