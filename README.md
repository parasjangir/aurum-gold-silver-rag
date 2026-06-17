# Aurum 🪙

**Gold & Silver Intelligence** — a Retrieval-Augmented Generation (RAG) chatbot
that answers anything about **gold & silver in India** (purity, hallmarking,
making charges, GST, rates/bhav, pricing, investment) — grounded in a knowledge
base, **citing its sources**, and computing real prices from the day's bhav.

Built from scratch as a learning project: ingestion → embeddings → vector search
→ grounded generation → evaluated with quality gates.

> The project folder is `sona-rag/` and the vector collection is `sonarag` — the
> product was renamed to **Aurum**; the internal names are kept to avoid churn.

## What it can do

- **Ask anything gold/silver:** karat & purity, BIS hallmarking & HUID, making
  charges & wastage, GST, gold rate/bhav, full price calculations, investment
  (jewellery vs coins vs SGB vs ETF), silver & sterling.
- **Live rates (GoldAPI.io):** auto-fetches live per-gram gold (24/22/18/14K) &
  silver rates; refresh button + manual override in the sidebar.
- **Exact pricing, guaranteed:** price queries are computed in **Python**, not by
  the LLM (rate + making% → ×weight → +3% GST). Precise every time, in any
  language, and works even if the LLM quota is down.
- **Grounded + cited:** India-specific facts come from the knowledge base and are
  cited as `[n]`; honest about general vs verified info.
- **Multilingual:** ask in **English · हिन्दी · Hinglish · मारवाड़ी** — Aurum
  replies in the same language (multilingual embeddings + language-aware prompt).
- **Premium UI:** dark glassmorphism, animated gold shimmer, aura glow, hover
  light-sweeps, live rates strip — switch language from the sidebar.
- **Stays in its lane:** politely declines non-precious-metal questions.
- **Evaluated:** a quality-gated eval harness (`evaluate.py`) guards regressions.

## Architecture

```
docs ─▶ CHUNK ─▶ EMBED ─▶ VECTOR DB ─┐   question ─▶ EMBED ─▶ similarity search
                                     └──────────────────────────────┘
                          today's bhav ─┐        top-k passages
                                        ▼               ▼
                              Claude/Groq LLM + rates + context
                                        │
                                        ▼
                          grounded answer + price math + [citations]
```

## Tech stack

| Layer | Choice |
|---|---|
| Embeddings | `sentence-transformers` multilingual MiniLM (free, local, Hindi-ready) |
| Vector store | ChromaDB (persistent, cosine) |
| Generation | Groq — Llama 3.3 70B (free), swappable to Claude/Ollama via `config.LLM_BACKEND` |
| Rates | **GoldAPI.io** live spot (auto-synced to `rates.json`) + manual override |
| Pricing | deterministic Python (`rates.py`) — never LLM arithmetic |
| Interface | Streamlit (futuristic dark + gold UI) |

## Status

- [x] Phases 0–6 — scaffold · chunking · embeddings · retrieval · generation+citations · UI · eval
- [x] **Upgrade** — expert behavior, live bhav + price calc, futuristic UI, rebrand to Aurum
- [ ] Phase 7 — polish & deploy (Streamlit Community Cloud)

Quality: **24 tests pass**; eval gates — Recall@5 1.00 · Refusal 1.00 · Citation 0.89 · Grounding 0.89.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # paste your free Groq key (console.groq.com)
```

## Run

```bash
streamlit run app.py          # the Aurum web app
python rag.py                 # terminal chat
python evaluate.py            # eval harness + quality gates
python -m pytest -q           # test suite (24 tests; some skip without a key)
```

## ⚠️ Notes

- `knowledge/` docs are **educational samples** — replace with authoritative BIS /
  GST text before real use.
- `rates.json` ships with **sample rates** — set your real daily bhav in the
  sidebar (or edit the file).
- Your `.env` (the API key) is gitignored; on Streamlit Cloud use their Secrets.
