import json
import os
from pathlib import Path

import pytest

from tests.fake_llm import FakeLLM

GOLDEN_PATH = Path(__file__).parent / "golden" / "tickets.json"
SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_tickets.json"


def load_golden() -> list[dict]:
    return json.loads(GOLDEN_PATH.read_text())


def load_samples() -> dict[str, dict]:
    return {t["id"]: t for t in json.loads(SAMPLE_PATH.read_text())}


GOLDEN = load_golden()
SAMPLES = load_samples()


def _live_enabled() -> bool:
    return os.getenv("RUN_LIVE_EVALS") == "1"


@pytest.fixture(autouse=True)
def fake_llm(request, monkeypatch):
    """Every offline test runs against the fake; live tests hit real providers."""
    if request.node.get_closest_marker("live"):
        if not _live_enabled():
            pytest.skip("set RUN_LIVE_EVALS=1 (and provider keys) to run live evals")
        yield None
        return

    fake = FakeLLM(GOLDEN)
    fake.install(monkeypatch)
    yield fake
    assert not fake.unexpected, (
        f"Unrecognised LLM/network endpoints called: {sorted(set(fake.unexpected))}. "
        "Add a route in tests/fake_llm.py if this provider is intentional."
    )


@pytest.fixture
def golden():
    return {t["id"]: t for t in GOLDEN}


@pytest.fixture
def samples():
    return SAMPLES
