"""Phase 2b — Vector store (ChromaDB).

A vector store holds our chunk vectors and answers the question: "given this
query vector, which stored vectors are closest?" That nearest-neighbour search
is the 'Retrieval' in Retrieval-Augmented Generation.

We use ChromaDB in PERSISTENT mode: it writes an index to disk (config.VECTOR_DIR)
so we build it once and reuse it across runs, instead of re-embedding every time.

WHY WE PASS OUR OWN EMBEDDINGS
------------------------------
Chroma can embed text for you, but we hand it vectors from OUR multilingual
model (embeddings.py). The rule from Phase 2a holds: documents and queries must
be embedded by the same model, so we control both ends.

The collection is configured for COSINE distance. Chroma returns a *distance*
(0 = identical), so we convert it to a *similarity* (1 = identical) for display:
similarity = 1 - distance.
"""
from __future__ import annotations

import chromadb

import config
import embeddings as emb
from chunking import Chunk, chunk_documents

COLLECTION_NAME = "sonarag"


def _client() -> chromadb.ClientAPI:
    config.VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.VECTOR_DIR))


def get_collection():
    """Get (or create) the collection, configured for cosine similarity."""
    return _client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def build_index(chunks: list[Chunk] | None = None) -> int:
    """Embed every chunk and store it in the vector DB. Returns the row count.

    Uses `upsert` (keyed on chunk.id) so re-running rebuilds in place instead of
    creating duplicates.
    """
    if chunks is None:
        chunks = chunk_documents()

    collection = get_collection()
    vectors = emb.embed_texts([c.text for c in chunks])

    collection.upsert(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=vectors,
        metadatas=[
            {
                "source": c.source,
                "chunk_index": c.chunk_index,
                "char_start": c.char_start,
                "char_end": c.char_end,
            }
            for c in chunks
        ],
    )
    return collection.count()


def search(query: str, k: int = 4) -> list[dict]:
    """Return the k chunks most similar in meaning to `query`."""
    collection = get_collection()
    query_vector = emb.embed_query(query)

    result = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    hits: list[dict] = []
    for doc, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append(
            {
                "text": doc,
                "source": meta["source"],
                "chunk_index": meta["chunk_index"],
                "similarity": round(1 - dist, 3),  # cosine distance -> similarity
            }
        )
    return hits


if __name__ == "__main__":
    count = build_index()
    print(f"Index built: {count} chunks stored in {config.VECTOR_DIR}\n")

    demo_queries = [
        "What does a BIS hallmark contain?",
        "How do I calculate the pure gold in a chain?",   # no keyword 'fineness'/'karat'
        "सोने की शुद्धता कैसे मापी जाती है?",              # Hindi: how is gold purity measured?
    ]
    for q in demo_queries:
        print(f"Q: {q}")
        for hit in search(q, k=2):
            preview = hit["text"][:90].replace("\n", " ")
            print(f"   [{hit['similarity']:.3f}] {hit['source']}#{hit['chunk_index']}: {preview}...")
        print()
