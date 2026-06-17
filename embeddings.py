"""Phase 2a — Embeddings.

WHAT IS AN EMBEDDING?
---------------------
An embedding is a list of numbers (a vector) that captures the *meaning* of a
piece of text. Texts with similar meaning get vectors that point in similar
directions, even if they share no words:

    "How pure is 22 karat gold?"   ─┐
    "What fineness is 22K?"         ─┴─▶  nearly the same vector

That's the magic that lets us search by meaning instead of keywords. A keyword
search for "fineness" would miss a chunk that only says "purity"; an embedding
search finds it because the two are close in vector space.

HOW WE MEASURE CLOSENESS
------------------------
Cosine similarity = the cosine of the angle between two vectors. 1.0 = identical
direction (same meaning), 0 = unrelated. We ask the model for *normalised*
(unit-length) vectors so cosine similarity is just their dot product, and the
vector DB can compare them efficiently.

THE MODEL
---------
We use a local `sentence-transformers` model (no API, no cost). It's
MULTILINGUAL, so a Hindi or Marwari question can still match an English chunk —
that's the edge we're building in.
"""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

import config


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    """Load the embedding model once and reuse it (it's heavy to construct)."""
    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts into a list of unit-length vectors."""
    model = _model()
    vectors = model.encode(
        list(texts),
        normalize_embeddings=True,   # unit length -> cosine similarity = dot product
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query string. (Same model as the documents — this matters:
    you must embed queries and documents with the *same* model, or the vectors
    live in different spaces and similarity is meaningless.)"""
    return embed_texts([text])[0]


if __name__ == "__main__":
    # Tiny demo: show that meaning beats keywords.
    a = embed_query("How pure is 22 karat gold?")
    b = embed_query("What fineness is 22K gold?")          # same meaning, diff words
    c = embed_query("What are the making charges on a necklace?")  # different topic

    def cosine(u, v):
        return sum(x * y for x, y in zip(u, v))  # both are unit vectors

    print(f"Embedding dimension: {len(a)}")
    print(f"similar meaning   (22K purity vs fineness): {cosine(a, b):.3f}")
    print(f"different topic   (22K purity vs charges) : {cosine(a, c):.3f}")
