"""Tests for Phase 2 (embeddings + vector store).

These load the embedding model and build a real index, so they're slower than
the chunking tests. Run from the project root with the venv active: pytest
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from chunking import chunk_documents
import vector_store as vs


@pytest.fixture(scope="module")
def index_count():
    """Build the index once for all tests in this module."""
    return vs.build_index()


def test_index_has_every_chunk(index_count):
    assert index_count == len(chunk_documents())


def test_search_returns_k_hits(index_count):
    hits = vs.search("What is a BIS hallmark?", k=3)
    assert len(hits) == 3
    assert all("text" in h and "source" in h for h in hits)


def test_similarities_are_sorted_and_bounded(index_count):
    hits = vs.search("How is gold purity measured?", k=4)
    sims = [h["similarity"] for h in hits]
    assert sims == sorted(sims, reverse=True)        # best match first
    assert all(-1.0 <= s <= 1.0 for s in sims)       # valid cosine range


def test_semantic_retrieval_finds_the_right_document(index_count):
    # Semantic match: the phrasing doesn't keyword-overlap the purity doc heavily,
    # yet meaning-based search still ranks it top.
    hits = vs.search("how is gold purity measured?", k=3)
    sources = {h["source"] for h in hits}
    assert "02_purity_and_karat.md" in sources


def test_hallmark_query_retrieves_hallmark_doc(index_count):
    hits = vs.search("which marks appear when I read a hallmark?", k=3)
    sources = {h["source"] for h in hits}
    assert sources & {"01_hallmarking_bis.md", "04_reading_a_hallmark.md"}
