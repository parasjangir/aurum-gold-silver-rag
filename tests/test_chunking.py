"""Tests for Phase 1 chunking. Run from the project root with: pytest"""
import sys
from pathlib import Path

# Make the project root importable when pytest runs from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from chunking import Chunk, chunk_documents, chunk_text, load_documents


# A controlled synthetic document: 6 short paragraphs.
SAMPLE = "\n\n".join(f"Paragraph {i}. " + ("alpha " * 20).strip() for i in range(6))


def test_loads_real_knowledge_base():
    docs = load_documents()
    assert len(docs) >= 4                       # the four sample .md files
    assert all(text.strip() for text in docs.values())


def test_chunks_are_non_empty_and_sized():
    chunks = chunk_text(SAMPLE, source="sample.md", chunk_size=300, overlap=60)
    assert len(chunks) > 1                        # the sample is big enough to split
    assert all(c.text for c in chunks)            # no empty chunks
    assert all(isinstance(c, Chunk) for c in chunks)


def test_char_offsets_reconstruct_the_text():
    # The recorded offsets must point back at exactly the chunk's text.
    chunks = chunk_text(SAMPLE, source="sample.md", chunk_size=300, overlap=60)
    for c in chunks:
        assert SAMPLE[c.char_start:c.char_end].strip() == c.text


def test_chunks_overlap():
    chunks = chunk_text(SAMPLE, source="sample.md", chunk_size=300, overlap=60)
    # Consecutive chunks should share some span (char_start of next < char_end of prev).
    overlaps = [
        chunks[k].char_end - chunks[k + 1].char_start
        for k in range(len(chunks) - 1)
    ]
    assert any(o > 0 for o in overlaps), "expected at least one overlapping pair"


def test_ids_unique_within_document():
    chunks = chunk_documents()
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be unique"


def test_no_overlap_when_overlap_is_zero():
    chunks = chunk_text(SAMPLE, source="sample.md", chunk_size=300, overlap=0)
    for k in range(len(chunks) - 1):
        assert chunks[k].char_end <= chunks[k + 1].char_start
