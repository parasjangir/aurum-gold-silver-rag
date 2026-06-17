# SonaRAG — Study Guide

Your companion notes for building SonaRAG zero → advanced. One lesson per phase.

---

## The big picture: what is RAG?

A plain LLM answers from memory (its training data). It can't know your private
documents, and it will confidently make things up ("hallucinate") when it
doesn't know.

**Retrieval-Augmented Generation** fixes both problems:

1. **Retrieve** — find the passages in *your* documents most relevant to the question.
2. **Augment** — paste those passages into the prompt as context.
3. **Generate** — ask the LLM to answer *using only that context*, and cite it.

The model stops guessing and starts quoting your sources. That's the whole idea.

---

## Lesson 1 — Ingestion & Chunking  ✅

### Why chunk?
We never hand the LLM whole documents. We split them into small, overlapping
passages ("chunks") because:
- the context window is finite and large contexts cost money + add noise;
- retrieval is far more precise on focused passages than on whole files.

### The three dials
| Dial | Too small | Too big |
|---|---|---|
| **Chunk size** | context gets lost, facts fragment | noisy retrieval, wasted tokens |
| **Overlap** | a fact on a boundary is split & lost | redundant storage |
| **Structure** | (we split on paragraphs so we don't cut sentences) | |

Sweet spot for this project: ~800 chars (~200 tokens), 120-char overlap. All
set in `config.py` so you can experiment in one place.

### How `chunking.py` works
1. `load_documents()` reads every `.md`/`.txt` in `knowledge/`.
2. `_split_into_blocks()` splits on blank lines → paragraph blocks.
3. `chunk_text()`:
   - **Step 1**: greedily pack consecutive paragraphs into windows ≤ chunk_size.
   - **Step 2**: give each window (after the first) the last ~120 chars of the
     previous one as an overlap prefix, snapped to a word boundary.
4. Every `Chunk` records `source`, `chunk_index`, and `char_start/char_end` —
   the metadata that powers **source citations** in Phase 4.

### The bug we fixed (worth remembering)
My first overlap attempt re-included whole paragraph *blocks*. When a paragraph
was bigger than the overlap target, neighbouring chunks ended up with almost no
overlap (24 chars instead of 120). A unit test caught it. **Lesson: measure
overlap in characters, not in whole structural units — and write the test that
proves your invariant holds.**

### Token vs character note
We size chunks in characters (1 token ≈ 4 chars). For exact token counts we'll
use Claude's `count_tokens` endpoint later — never `tiktoken`, which is OpenAI's
tokenizer and miscounts for Claude.

### Exercises
1. In `config.py`, set `CHUNK_SIZE = 300` and re-run `python3 chunking.py`.
   How many chunks now? What happened to avg size?
2. Set `CHUNK_OVERLAP = 0`. Re-run the demo — confirm the reported overlap is 0.
   Then run `pytest` — which test now guards this behaviour?
3. **Harder:** a single paragraph longer than `chunk_size` is emitted whole
   (not sub-split). Find where in `chunk_text()` that happens, and sketch how
   you'd hard-split an oversized block on sentence boundaries.

### ✅ Check yourself before Phase 2
- Why do we add overlap between chunks?
- What metadata does a `Chunk` carry, and why will we need it for citations?
- If retrieval later returns irrelevant passages, name two chunking dials you'd
  reach for first.

---

## Lesson 2 — Embeddings & Vector Store  ✅

### What's an embedding?
A vector (list of numbers) that captures the *meaning* of text. Similar meanings
→ vectors pointing in similar directions, even with zero shared words. We saw it
live: "22K purity" vs "22K fineness" scored **0.859**; an unrelated topic scored
**0.412**. That gap is why semantic search beats keyword search.

### Cosine similarity
The cosine of the angle between two vectors. 1.0 = same direction (same meaning),
0 = unrelated. We ask the model for **normalised** (unit-length) vectors, so
cosine similarity = a simple dot product, and the DB can compare fast.

### The golden rule
**Embed queries and documents with the SAME model.** Different models = different
vector spaces = meaningless comparisons. `embeddings.py` is the single source of
truth for both.

### What `embeddings.py` does
- Loads a multilingual `sentence-transformers` model once (`@lru_cache`).
- `embed_texts(list)` / `embed_query(str)` → unit-length 384-dim vectors.
- Multilingual = a Hindi/Marwari question can match an English chunk. Proven live:
  a Hindi purity question retrieved the English purity doc at 0.640.

### What `vector_store.py` does (ChromaDB)
- **Persistent** store on disk (`vector_store/`) — build once, reuse across runs.
- `build_index()` embeds every chunk and `upsert`s it with its metadata
  (`source`, `chunk_index`, `char_start/end`) — keyed on `chunk.id`, so re-running
  rebuilds in place, no duplicates.
- `search(query, k)` embeds the query, finds the k nearest vectors, and returns
  text + source + a **similarity** score (we convert Chroma's cosine *distance*
  via `similarity = 1 - distance`).

### Mental model
```
chunks ─embed─▶ [vectors] ─store─▶ ChromaDB
                                      ▲
query ─embed─▶ [vector] ─nearest-k───┘ ─▶ top passages (+ metadata for citations)
```

### Exercises
1. Run `python vector_store.py`. Pick one result and explain *why* it ranked #1
   for that question even though it may not share the question's exact words.
2. In `vector_store.py`'s demo, add a query about **GST/making charges** and check
   it retrieves `03_gst_and_pricing.md` (not the purity doc). What similarity did
   the top hit get?
3. **Harder:** the index is rebuilt every time `vector_store.py` runs. That's fine
   for 13 chunks, but wasteful for thousands. Sketch how you'd skip re-embedding
   chunks whose text hasn't changed (hint: `chunk.id` + a content hash).

### Check yourself before Phase 3
- Why must queries and documents share the same embedding model?
- Chroma gives a *distance*; how do we turn it into a *similarity*, and which is "better" when higher?
- Where do the `source` + char offsets in each stored record get used later?

---

## Lesson 3 — Retrieval  ✅

### The problem with raw search
`search()` always returns the k *closest* chunks — but "closest" ≠ "relevant".
Ask about the weather and you still get the least-irrelevant gold chunk. Feed
that to an LLM and it hallucinates an answer from noise.

### The fix: a measured threshold
We added `SIMILARITY_THRESHOLD = 0.35` and drop any hit below it. The number
isn't a guess — we **measured** the score distribution first:

| | top similarity |
|---|---|
| on-topic questions | 0.63 – 0.71 |
| off-topic questions | 0.02 – 0.09 |

A wide canyon between them, so 0.35 separates relevant from noise with margin.
**Always set thresholds from data, not vibes.** (Re-measure if you change the
embedding model or the documents — the numbers shift.)

### `retriever.py`
- `retrieve(query, k, threshold)` → a `RetrievedContext`.
- `RetrievedContext.found` — did anything clear the bar? If not, the bot will
  honestly say "I don't have info on that" (Phase 4). **First anti-hallucination
  guardrail.**
- `.context` — survivors packaged as a numbered, source-labelled block:
  ```
  [1] (source: 01_hallmarking_bis.md)
  ...passage text...

  [2] (source: 04_reading_a_hallmark.md)
  ...
  ```
- `.citations` — `{1: '01_hallmarking_bis.md', ...}`. Phase 4 turns the `[1]`
  markers the model uses into a real source list.

### Why the numbering matters
We give the model passages tagged `[1] [2] [3]`. In Phase 4 we'll instruct it:
"cite the passage number you used." That's how a generated sentence gets tied
back to a real source — engineering maturity that recruiters notice.

### Exercises
1. In `config.py`, drop `SIMILARITY_THRESHOLD` to `0.05` and re-run
   `python retriever.py` with the off-topic weather query. What comes back now,
   and why is that bad?
2. Raise `TOP_K` to 8. Does the on-topic query keep more passages? Are the extra
   ones still relevant, or is some noise sneaking in just under the old `k`?
3. **Harder:** all 4 kept passages for the hallmark query came from the same 1-2
   files. For a question spanning topics (e.g. "how is a hallmarked 22K ring
   priced?"), would top-k from one similarity score pull from *both* the
   hallmark and the pricing docs? Sketch when you'd want per-source diversity.

### Check yourself before Phase 4
- Why is "closest chunk" not the same as "relevant chunk"?
- How did we choose 0.35, and what should you do after swapping the embedding model?
- What two things does `RetrievedContext` give Phase 4 — one for the prompt, one for the citations?

---

## Lesson 4 — Generation + Citations  ✅ (built; runs live once a key is set)

### The payoff: RAG fully assembled
```
Retrieve (Phase 3) → Augment (grounded prompt) → Generate (LLM) → cite [n]
```

### The LLM is a SWAPPABLE component
`llm.py` exposes one function — `generate(prompt, system) -> str` — and the rest
of the project calls only that. The backend is chosen in `config.LLM_BACKEND`
("groq" / "claude" / "ollama"). **Why this matters:** we picked Groq today
because it's free with no card; if you get an Anthropic key later, switching to
Claude is a one-line config change, not a rewrite. Good systems isolate the
parts most likely to change.

### Two things make the answers trustworthy
1. **Guardrail runs first.** In `rag.answer()`, if retrieval found nothing above
   the threshold, we return "I don't know" *without calling the LLM*. No context
   → no answer → no hallucination → no wasted tokens. (Tested without any key.)
2. **The prompt is grounded.** The system prompt orders the model to: answer
   ONLY from the numbered passages, cite the `[n]` used, and refuse if the answer
   isn't there. Grounding + citation is the line between a real RAG system and a
   confident bullshitter.

### How citations work end to end
Phase 3 labelled passages `[1] (source: file.md)`. Here we tell the model to cite
`[n]`, and `Answer.sources` maps each number back to its file. So a sentence in
the answer → `[2]` → `02_purity_and_karat.md`. That traceability is the
"engineering maturity" feature recruiters notice.

### Why low temperature
`TEMPERATURE = 0.2`. We want faithful summarisation of the passages, not
creative variation. High temperature = more invention = more risk.

### Backend choice (Groq) on an 8 GB Mac
Local 7B+ models are tight on 8 GB and you're already running torch+Chroma for
embeddings. Groq runs big models in the cloud for free (no card), so it's the
better fit for a demo. "API key" ≠ "paid" — Groq/Gemini issue free keys.

### Exercises
1. With your key set, run `python rag.py` and ask: *"How is a 10g 22K chain's
   pure gold weight calculated?"* Check the answer cites a passage and matches
   `03_gst_and_pricing.md` / `02_purity_and_karat.md`.
2. Ask something the docs DON'T cover (e.g. *"What's the gold rate today?"*).
   Confirm SonaRAG refuses instead of inventing a number.
3. **Harder:** open `rag.py`, weaken the system prompt (remove the "ONLY use the
   passages" line), and ask an off-topic-but-tempting question. Watch grounding
   degrade. Put it back. This shows how much of RAG quality lives in the prompt.

### Check yourself before Phase 5
- Why does the guardrail run *before* the LLM call, not after?
- Name one concrete benefit of hiding the LLM behind `llm.py`.
- What turns a `[2]` in the answer text into a named source file?

---

## Lesson 5 — Interface (Streamlit)  ✅

### What changed
Nothing in the RAG logic. `app.py` is *pure presentation* — it calls
`rag.answer()` and renders the result. Keeping the UI as a thin layer over a
tested core is the right separation: you can swap Streamlit for a FastAPI
endpoint or a CLI without touching retrieval or generation.

### Streamlit ideas you used
- **`st.chat_message` / `st.chat_input`** — the built-in chat primitives.
- **`st.session_state.history`** — Streamlit re-runs the whole script on every
  interaction, so any state you want to persist (the conversation) lives in
  `session_state`, not local variables.
- **`@st.cache_resource`** — the embedding model + index build are expensive, so
  we memoise them once per server process instead of rebuilding on every rerun.
- **`st.expander("📎 Sources")`** — the citation payoff, made visible: each
  answer's `[n]` markers map to source files in a collapsible panel.

### The rerun mental model (the thing that trips everyone up)
Streamlit isn't event-driven like a normal web app. On *every* click or input,
it runs `app.py` top to bottom again. So you: (1) read input, (2) mutate
`session_state`, (3) render the whole UI from `session_state`. If something
"resets" unexpectedly, it's almost always because you stored it in a plain
variable instead of `session_state`.

### Run it
```bash
streamlit run app.py
```
Then click an example or type a question. Try an off-topic one and watch it
refuse — the guardrail shows up in the UI too.

### Exercises
1. Add a slider in the sidebar for `TOP_K` (1–8) and pass it to `answer()`. Watch
   how more/fewer passages change the answer. (Hint: `st.slider`.)
2. The Sources panel lists files but not the passage text. Extend `Answer` to
   also carry the retrieved hits, and show each cited passage's snippet under its
   number — full transparency.
3. **Harder:** right now the embedding model loads on first request (slow first
   answer). Read about Streamlit's startup and decide where you'd warm it.

### Check yourself before Phase 6
- Why does conversation state need to live in `st.session_state`?
- What does `@st.cache_resource` save us from doing on every interaction?
- `app.py` imports `rag.answer` and nothing about embeddings/Chroma directly. Why
  is that a good sign?

---

## Lesson 6 — Evaluation & Guardrails  ✅

### Why bother — "it worked when I tried it" is not evidence
Anyone can cherry-pick one good answer. An eval harness turns quality into
numbers you can defend in an interview and re-run after every change to catch
regressions. `evaluate.py` + `eval/eval_set.json` do exactly that.

### We measure the two layers separately (they fail for different reasons)
**Retrieval** (no LLM, deterministic — runs without a key):
- **Recall@k** — is an expected source among the top-k retrieved chunks?
- **MRR** — how high did the first correct source rank? (1/rank)

**Generation** (live, needs the key):
- **Refusal accuracy** — does it refuse the unanswerable questions?
- **Citation rate** — do answerable answers contain a `[n]`?
- **Keyword grounding** — does the answer contain the expected fact?

### Quality gates = a regression alarm
`GATES` sets a floor per metric; `evaluate.py` exits non-zero if any metric
falls below it. Wire that into CI and a bad chunking/model/threshold change fails
the build instead of silently shipping.

### What the eval CAUGHT (this is the whole point)
1. **Citation rate 0.50.** Retrieval, refusal, grounding were all 1.00 — but the
   model only cited half the time. We **fixed the prompt** (firmer rule + an
   explicit format example), re-ran → **1.00**. Fix the system, not the test.
2. **A bug in our own metric.** `has_citation` used `\[\d+\]`, which misses
   `[1, 3]` — it would *undercount* citations. A unit test caught it; we widened
   the regex. Even your measuring tools need tests.

Both are exactly the kind of thing you mention in an interview: "my eval surfaced
a citation gap and a metric bug, here's how I fixed each."

### Results (current)
```
Recall@4 1.00 · MRR 1.00 · Refusal 1.00 · Citation 1.00 · Grounding 1.00
✅ ALL GATES PASSED   (23 unit tests green)
```

### Exercises
1. Add a question whose answer ISN'T in the docs but sounds like it should be
   (e.g. "what's the hallmarking fee in rupees?"). Mark it `refuse`. Does the
   system pass? If not, is that a retrieval or a generation problem?
2. Tighten `GATES["recall_at_k"]` to 1.0 and add a question the retriever gets
   wrong, to watch a gate fail on purpose.
3. **Harder:** our grounding check is keyword-based. Sketch an *LLM-as-judge*
   faithfulness check: feed the passages + the answer to the model and ask
   "is every claim supported? yes/no." What are the trade-offs vs keywords?

### Check yourself before Phase 7
- Why measure retrieval and generation separately?
- What does a quality gate give you that a one-off manual test doesn't?
- The eval found citation rate was low while grounding was perfect. In plain
  English, what was wrong — and where did we fix it?

---

## Lesson 7 — Polish & Deploy  (next)
Ship it: tidy the README with a demo, and deploy the Streamlit app to Streamlit
Community Cloud so you have a public link for your CV/portfolio. Coming up.
