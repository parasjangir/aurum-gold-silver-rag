"""Phase 1 — Document ingestion & chunking.

THE PROBLEM
-----------
We have documents (about gold, hallmarking, GST...). A user asks a question.
We want to feed *only the relevant passages* to the LLM, not entire documents,
because:
  1. An LLM's context window is finite (and big contexts cost money + add noise).
  2. Retrieval works far better on small, focused passages than on whole files.

So before anything else, we split each document into "chunks" — small,
overlapping windows of text. This file does exactly that, using nothing but
the Python standard library.

KEY IDEAS YOU'RE LEARNING HERE
------------------------------
* Chunk SIZE   — too big = noisy retrieval & wasted tokens; too small = lost
                 context. ~200-300 tokens is a common sweet spot.
* OVERLAP      — neighbouring chunks share some text so a fact sitting on a
                 boundary survives in at least one chunk.
* STRUCTURE    — we split on blank lines (paragraph boundaries) first, so we
                 don't slice a sentence in half. Then we PACK paragraphs
                 together until we hit the size budget.
* METADATA     — every chunk remembers WHERE it came from (file + offsets).
                 This is what makes source citations possible in Phase 4.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import config


@dataclass
class Chunk:
    """One retrievable passage, plus the metadata that lets us cite it later."""
    text: str
    source: str          # filename the chunk came from, e.g. "02_purity_and_karat.md"
    chunk_index: int     # 0-based position of this chunk within its document
    char_start: int      # offset in the original document (inclusive)
    char_end: int        # offset in the original document (exclusive)

    @property
    def id(self) -> str:
        """A stable, unique identifier — used as the key in the vector DB."""
        return f"{self.source}::chunk-{self.chunk_index}"


def load_documents(knowledge_dir: Path = config.KNOWLEDGE_DIR) -> dict[str, str]:
    """Read every .md / .txt file in the knowledge directory into {filename: text}."""
    docs: dict[str, str] = {}
    for path in sorted(knowledge_dir.glob("*")):
        if path.suffix.lower() in {".md", ".txt"}:
            docs[path.name] = path.read_text(encoding="utf-8")
    if not docs:
        raise FileNotFoundError(f"No .md/.txt documents found in {knowledge_dir}")
    return docs


def _split_into_blocks(text: str) -> list[str]:
    """Split text on blank lines into paragraph-sized blocks.

    Splitting on structure first means a chunk boundary lands *between*
    paragraphs, never in the middle of a sentence.
    """
    blocks = re.split(r"\n\s*\n", text.strip())
    return [b.strip() for b in blocks if b.strip()]


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split one document into overlapping, paragraph-aware chunks.

    Algorithm (two clean steps):
      STEP 1 — Partition the doc into contiguous "windows". Break it into
        paragraph blocks, then greedily pack consecutive blocks into a window
        until the next block would bust `chunk_size`. Windows don't overlap yet
        and never cut a paragraph in half.
      STEP 2 — Add overlap. Each window after the first also carries the last
        ~`overlap` characters of text that came before it. We snap that prefix
        forward to a word boundary so it doesn't start mid-word.
    """
    blocks = _split_into_blocks(text)

    # Record where each block lives in the original text, so char offsets are real.
    spans: list[tuple[int, int]] = []
    cursor = 0
    for block in blocks:
        start = text.index(block, cursor)
        spans.append((start, start + len(block)))
        cursor = start + len(block)

    # STEP 1: greedy paragraph packing -> a list of non-overlapping (start, end).
    windows: list[tuple[int, int]] = []
    i, n = 0, len(spans)
    while i < n:
        win_start = spans[i][0]
        j = i
        while j + 1 < n and spans[j + 1][1] - win_start <= chunk_size:
            j += 1
        windows.append((win_start, spans[j][1]))
        i = j + 1  # next window starts after this one — guaranteed progress

    # STEP 2: build chunks, prepending the overlap prefix to all but the first.
    chunks: list[Chunk] = []
    for idx, (start, end) in enumerate(windows):
        real_start = start
        if idx > 0 and overlap > 0:
            # back up `overlap` chars, but not past the previous window's start
            real_start = max(windows[idx - 1][0], start - overlap)
            # snap forward to just after a space so we don't begin mid-word
            sp = text.find(" ", real_start, start)
            if sp != -1 and sp + 1 < start:
                real_start = sp + 1
        chunk_str = text[real_start:end].strip()
        chunks.append(Chunk(chunk_str, source, idx, real_start, end))

    return chunks


def chunk_documents(knowledge_dir: Path = config.KNOWLEDGE_DIR) -> list[Chunk]:
    """Load and chunk every document in the knowledge base."""
    docs = load_documents(knowledge_dir)
    all_chunks: list[Chunk] = []
    for name, text in docs.items():
        all_chunks.extend(chunk_text(text, source=name))
    return all_chunks


if __name__ == "__main__":
    # A small demo so you can SEE what chunking produced.
    chunks = chunk_documents()
    docs = load_documents()

    sizes = [len(c.text) for c in chunks]
    print(f"Documents loaded : {len(docs)}")
    print(f"Chunks produced  : {len(chunks)}")
    print(f"Avg chunk size   : {sum(sizes) // len(sizes)} chars")
    print(f"Min / Max size   : {min(sizes)} / {max(sizes)} chars")
    print("-" * 70)

    # Show the first two chunks of the first document to make overlap visible.
    first_doc = next(iter(docs))
    doc_chunks = [c for c in chunks if c.source == first_doc][:2]
    for c in doc_chunks:
        print(f"\n[{c.id}]  chars {c.char_start}-{c.char_end}")
        preview = c.text[:200].replace("\n", " ")
        print(f"  {preview}{'...' if len(c.text) > 200 else ''}")

    if len(doc_chunks) == 2:
        a, b = doc_chunks
        overlap_chars = max(0, a.char_end - b.char_start)
        print("-" * 70)
        print(f"Overlap between chunk 0 and chunk 1: {overlap_chars} chars "
              f"(target was {config.CHUNK_OVERLAP})")
