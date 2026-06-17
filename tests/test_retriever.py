"""Tests for Phase 3 (retrieval + threshold guardrail)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import config
import vector_store as vs
from retriever import retrieve


@pytest.fixture(scope="module", autouse=True)
def _ensure_index():
    vs.build_index()


def test_on_topic_query_finds_context():
    r = retrieve("What are the three marks on a BIS hallmark?")
    assert r.found
    assert len(r.hits) >= 1
    # every kept hit must clear the threshold
    assert all(h["similarity"] >= config.SIMILARITY_THRESHOLD for h in r.hits)


def test_off_topic_query_returns_nothing():
    r = retrieve("What's the weather in Mumbai today?")
    assert not r.found
    assert r.hits == []
    assert r.context == ""


def test_context_is_numbered_and_source_labelled():
    r = retrieve("How is gold purity expressed as fineness?")
    assert "[1]" in r.context
    assert "(source:" in r.context
    # the first citation's source must appear in the context block
    assert r.citations[1] in r.context


def test_citations_match_hits():
    r = retrieve("What is sterling silver?")
    assert len(r.citations) == len(r.hits)
    for i, source in r.citations.items():
        assert r.hits[i - 1]["source"] == source


def test_threshold_is_respected():
    # An impossibly high threshold should reject even an on-topic query.
    r = retrieve("What are the three marks on a BIS hallmark?", threshold=0.99)
    assert not r.found
