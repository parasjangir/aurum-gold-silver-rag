"""Tests for the RAG pipeline (Aurum).

The off-domain guard runs ALWAYS (deterministic, no LLM). The live answer test
runs only when GROQ_API_KEY is set.
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import pytest

import vector_store as vs
from rag import OFF_DOMAIN_MESSAGE, answer


@pytest.fixture(scope="module", autouse=True)
def _ensure_index():
    vs.build_index()


def test_off_domain_is_refused_without_calling_the_llm():
    # Below the similarity floor -> deterministic refusal, no LLM call.
    r = answer("What's the weather in Mumbai today?")
    assert r.found is False
    assert r.text == OFF_DOMAIN_MESSAGE
    assert r.sources == {}


def test_price_query_is_deterministic_and_exact():
    # Price answers are computed in Python — exact, and no API key needed.
    from rates import estimate_price, load_rates

    r = answer("price of 12 grams 22K gold with 11% making charges?")
    assert r.found is True
    expected = estimate_price(12, karat=22, making_pct=11, rates=load_rates())["total"]
    assert f"{expected:,.2f}" in r.text


HAS_KEY = bool(os.getenv("GROQ_API_KEY"))


@pytest.mark.skipif(not HAS_KEY, reason="set GROQ_API_KEY to run live generation")
def test_in_domain_answer_is_grounded_and_cited():
    r = answer("What are the three marks on a BIS hallmark?")
    assert r.found is True
    assert r.text.strip()
    assert r.sources
    cited = set()
    for group in re.findall(r"\[([\d,\s]+)\]", r.text):
        cited.update(int(t) for t in group.split(",") if t.strip().isdigit())
    assert cited & set(r.sources), f"no valid citation in: {r.text!r}"


@pytest.mark.skipif(not HAS_KEY, reason="set GROQ_API_KEY to run live generation")
def test_gold_rate_question_is_answered_not_refused():
    # "Gold rate" is in-domain now — it must answer (from rates), not refuse.
    r = answer("What is today's 22K gold rate?")
    assert r.found is True
    assert OFF_DOMAIN_MESSAGE not in r.text
