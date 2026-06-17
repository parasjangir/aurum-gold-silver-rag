"""Phase 3 — Retrieval.

Phase 2 gave us `search()`, which always returns the k closest chunks. But
"closest" is not the same as "relevant": ask about the weather and you'll still
get back the least-irrelevant gold chunk. If we feed that to the LLM, it will
dutifully try to answer from junk — that's how hallucinations happen.

Phase 3 adds judgment on top of search:

  1. TOP-K          — pull a handful of candidates (config.TOP_K).
  2. THRESHOLD      — drop any candidate below config.SIMILARITY_THRESHOLD.
                      If nothing survives, we honestly report "no relevant info"
                      instead of answering from noise. This is our first
                      anti-hallucination guardrail.
  3. CONTEXT        — package the survivors into a clean, numbered block with
                      source labels, ready to drop into the LLM prompt. The
                      numbering ([1], [2], ...) is what the model will cite in
                      Phase 4.
"""
from __future__ import annotations

from dataclasses import dataclass

import config
import vector_store as vs


@dataclass
class RetrievedContext:
    """The result of a retrieval: the surviving hits plus everything Phase 4 needs."""
    query: str
    hits: list[dict]      # each: {text, source, chunk_index, similarity}

    @property
    def found(self) -> bool:
        """True if at least one chunk cleared the relevance threshold."""
        return bool(self.hits)

    @property
    def context(self) -> str:
        """The numbered, source-labelled passage block for the LLM prompt."""
        blocks = []
        for i, h in enumerate(self.hits, start=1):
            blocks.append(f"[{i}] (source: {h['source']})\n{h['text']}")
        return "\n\n".join(blocks)

    @property
    def citations(self) -> dict[int, str]:
        """Map each passage number to its source file, e.g. {1: '02_purity_and_karat.md'}."""
        return {i: h["source"] for i, h in enumerate(self.hits, start=1)}


def retrieve(
    query: str,
    k: int = config.TOP_K,
    threshold: float = config.SIMILARITY_THRESHOLD,
) -> RetrievedContext:
    """Search, then keep only the chunks that clear the relevance threshold."""
    candidates = vs.search(query, k=k)
    kept = [h for h in candidates if h["similarity"] >= threshold]
    return RetrievedContext(query=query, hits=kept)


if __name__ == "__main__":
    # Make sure the index exists (cheap upsert if already built).
    vs.build_index()

    on_topic = "What are the three marks on a BIS hallmark?"
    off_topic = "What's the weather in Mumbai today?"

    print(f"Q (on-topic):  {on_topic}")
    r = retrieve(on_topic)
    print(f"  found={r.found}  kept {len(r.hits)} passage(s)")
    print(f"  citations: {r.citations}")
    print("  --- context handed to the LLM ---")
    print("  " + r.context[:300].replace("\n", "\n  ") + " ...\n")

    print(f"Q (off-topic): {off_topic}")
    r = retrieve(off_topic)
    print(f"  found={r.found}  kept {len(r.hits)} passage(s)")
    if not r.found:
        print("  -> SonaRAG would reply: \"I don't have information about that in "
              "my knowledge base.\" (no hallucination)")
