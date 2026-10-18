"""Cost regression guard.

Runs each scenario against the fake, counts LLM calls and estimated input
tokens, and compares them with tests/baselines/llm_usage.json. A change may
lower usage freely but must not raise it. Results go to output/llm_usage.json so
a sandbox can diff before/after.

Re-record the baseline after an intentional change:
    UPDATE_LLM_BASELINE=1 pytest tests/test_llm_budget.py
"""
import importlib.util
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import SESSIONS, app
from app.pipeline import process_ticket
from app.services import kb_search
from tests.conftest import GOLDEN, SAMPLES

ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = Path(__file__).parent / "baselines" / "llm_usage.json"
REPORT_PATH = ROOT / "output" / "llm_usage.json"
TOKEN_TOLERANCE = 0.05
UPDATE = os.getenv("UPDATE_LLM_BASELINE") == "1"

_results: dict[str, dict] = {}


def _run_nightly():
    spec = importlib.util.spec_from_file_location("nightly_report", ROOT / "scripts" / "nightly_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def _run_chat():
    SESSIONS.clear()
    client = TestClient(app)
    for msg in ["My order SN-448812 says delivered but I never got it.", "It's been two days.", "Ok thanks."]:
        client.post("/chat", json={"session_id": "budget", "message": msg})


SCENARIOS = {
    **{f"pipeline:{t['id']}": (lambda tid=t["id"]: process_ticket(SAMPLES[tid])) for t in GOLDEN},
    "kb_search:single_query": lambda: kb_search.search(GOLDEN[0]["body_en"]),
    "chat:3_turns": _run_chat,
    "nightly_report": _run_nightly,
}


def _baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text()) if BASELINE_PATH.exists() else {}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_llm_usage_does_not_regress(name, fake_llm, capsys):
    SCENARIOS[name]()
    usage = fake_llm.usage()
    _results[name] = usage

    if UPDATE:
        pytest.skip("recording baseline")
    base = _baseline().get(name)
    if base is None:
        pytest.skip(f"no baseline for {name}; run with UPDATE_LLM_BASELINE=1")

    assert usage["calls"] <= base["calls"], f"{name}: {usage['calls']} LLM calls vs baseline {base['calls']}"
    limit = base["input_tokens"] * (1 + TOKEN_TOLERANCE)
    assert usage["input_tokens"] <= limit, (
        f"{name}: {usage['input_tokens']} input tokens vs baseline {base['input_tokens']}"
    )


def teardown_module(module):
    if not _results:
        return
    if UPDATE:
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(_results, indent=2, sort_keys=True) + "\n")

    base = _baseline()
    report = {}
    for name, usage in sorted(_results.items()):
        b = base.get(name, {})
        report[name] = {
            **usage,
            "baseline_calls": b.get("calls"),
            "baseline_input_tokens": b.get("input_tokens"),
            "calls_delta": usage["calls"] - b["calls"] if b else None,
            "input_tokens_delta": usage["input_tokens"] - b["input_tokens"] if b else None,
        }
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
