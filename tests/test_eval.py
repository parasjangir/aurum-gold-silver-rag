"""Tests for Phase 6 (evaluation harness).

Retrieval + helper tests run ALWAYS (no LLM). The generation-gate test runs
only when GROQ_API_KEY is set.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import pytest

import config
import vector_store as vs
from evaluate import (
    GATES,
    evaluate_generation,
    evaluate_retrieval,
    has_citation,
    load_eval_set,
    looks_like_refusal,
)


@pytest.fixture(scope="module", autouse=True)
def _ensure_index():
    vs.build_index()


def test_eval_set_schema():
    items = load_eval_set()
    assert len(items) >= 6
    for x in items:
        assert x["expect"] in {"answer", "refuse"}
        if x["expect"] == "answer":
            assert x["sources"] and isinstance(x["sources"], list)


def test_retrieval_meets_recall_gate():
    # Deterministic — proves the right document is retrievable without an API key.
    r = evaluate_retrieval(load_eval_set())
    assert r["recall_at_k"] >= GATES["recall_at_k"]
    assert r["mrr"] > 0.0


def test_refusal_helper():
    assert looks_like_refusal("anything", found=False) is True       # empty retrieval
    assert looks_like_refusal("I don't have that information.", True) is True
    assert looks_like_refusal("22K gold is 91.6% pure [1].", True) is False


def test_citation_helper():
    assert has_citation("Sterling silver is 92.5% [1].")
    assert has_citation("Both apply [1, 3].")
    assert not has_citation("No citation here.")


HAS_KEY = bool(os.getenv("GROQ_API_KEY"))


@pytest.mark.skipif(not HAS_KEY, reason="set GROQ_API_KEY to run the live generation gate")
def test_generation_meets_gates():
    g = evaluate_generation(load_eval_set())
    assert g["refusal_accuracy"] >= GATES["refusal_accuracy"]
    assert g["citation_rate"] >= GATES["citation_rate"]
    assert g["keyword_grounding"] >= GATES["keyword_grounding"]
