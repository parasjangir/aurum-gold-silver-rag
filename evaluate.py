"""Phase 6 — Evaluation & guardrails.

"It worked when I tried it" is not evidence. This module turns SonaRAG's quality
into NUMBERS we can re-run as a regression gate. We measure two layers
separately, because they fail for different reasons:

  RETRIEVAL  — did the right document come back at all? (no LLM, deterministic)
      * Recall@k : is an expected source among the top-k retrieved chunks?
      * MRR      : how high did the first correct source rank? (1/rank)

  GENERATION — given good retrieval, did the LLM behave? (needs the API key)
      * Refusal accuracy   : does it refuse the unanswerable questions?
      * Citation rate      : do answerable questions get a [n] citation?
      * Keyword grounding  : does the answer contain the expected fact?

At the end we check every metric against a GATE. If any gate fails, the script
exits non-zero — so this doubles as a CI check that catches regressions when you
change chunking, the model, or the threshold.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import config
import vector_store as vs
from rag import answer

EVAL_PATH = config.PROJECT_ROOT / "eval" / "eval_set.json"

# Quality gates — the run FAILS if any measured metric falls below these.
GATES = {
    "recall_at_k": 0.80,
    "refusal_accuracy": 1.00,
    "citation_rate": 0.80,
    "keyword_grounding": 0.80,
}

_REFUSAL_MARKERS = (
    "don't have", "do not have", "not contain", "no information",
    "don't know", "do not know", "couldn't find", "could not find",
)


def load_eval_set(path: Path = EVAL_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def looks_like_refusal(text: str, found: bool) -> bool:
    """A response counts as a refusal if retrieval was empty OR the text says so."""
    if not found:
        return True
    low = text.lower()
    return any(marker in low for marker in _REFUSAL_MARKERS)


def has_citation(text: str) -> bool:
    # Match [1], [1,3], and [1, 3] — any bracket containing a digit.
    return bool(re.search(r"\[\s*\d[\d\s,]*\]", text))


# --- Retrieval evaluation (no LLM, deterministic) --------------------------
def evaluate_retrieval(eval_set: list[dict], k: int = config.TOP_K) -> dict:
    items = [x for x in eval_set if x["expect"] == "answer"]
    recalls, rrs = [], []
    for x in items:
        ranked_sources = [h["source"] for h in vs.search(x["question"], k=k)]
        expected = set(x["sources"])
        recalls.append(1.0 if any(s in expected for s in ranked_sources) else 0.0)
        rr = 0.0
        for rank, s in enumerate(ranked_sources, start=1):
            if s in expected:
                rr = 1.0 / rank
                break
        rrs.append(rr)
    return {"recall_at_k": _mean(recalls), "mrr": _mean(rrs), "n": len(items), "k": k}


# --- Generation evaluation (needs GROQ_API_KEY) ----------------------------
def evaluate_generation(eval_set: list[dict]) -> dict:
    answerable = [x for x in eval_set if x["expect"] == "answer"]
    refusable = [x for x in eval_set if x["expect"] == "refuse"]

    refusal_correct, cited, grounded = [], [], []

    for x in refusable:
        a = answer(x["question"])
        refusal_correct.append(1.0 if looks_like_refusal(a.text, a.found) else 0.0)

    for x in answerable:
        a = answer(x["question"])
        refused = looks_like_refusal(a.text, a.found)
        cited.append(1.0 if (has_citation(a.text) and not refused) else 0.0)
        kws = x.get("must_include", [])
        kw_ok = any(kw.lower() in a.text.lower() for kw in kws) if kws else True
        grounded.append(1.0 if (kw_ok and not refused) else 0.0)

    return {
        "refusal_accuracy": _mean(refusal_correct),
        "citation_rate": _mean(cited),
        "keyword_grounding": _mean(grounded),
        "n_answer": len(answerable),
        "n_refuse": len(refusable),
    }


def _check_gates(metrics: dict) -> bool:
    print("\nGATES")
    all_pass = True
    for name, threshold in GATES.items():
        if name not in metrics:
            continue
        value = metrics[name]
        ok = value >= threshold
        all_pass &= ok
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {name:<18} {value:.2f}  (>= {threshold:.2f})")
    return all_pass


def main() -> int:
    vs.build_index()
    eval_set = load_eval_set()

    print("=" * 60)
    print("RETRIEVAL  (deterministic, no LLM)")
    r = evaluate_retrieval(eval_set)
    print(f"  Recall@{r['k']} : {r['recall_at_k']:.2f}   over {r['n']} answerable questions")
    print(f"  MRR       : {r['mrr']:.2f}")

    metrics = {"recall_at_k": r["recall_at_k"]}

    has_key = bool(__import__("os").getenv("GROQ_API_KEY"))
    if has_key:
        print("\n" + "=" * 60)
        print("GENERATION  (live, via the LLM)")
        g = evaluate_generation(eval_set)
        print(f"  Refusal accuracy  : {g['refusal_accuracy']:.2f}   over {g['n_refuse']} unanswerable")
        print(f"  Citation rate     : {g['citation_rate']:.2f}   over {g['n_answer']} answerable")
        print(f"  Keyword grounding : {g['keyword_grounding']:.2f}")
        metrics.update(g)
    else:
        print("\n(skipping GENERATION eval — set GROQ_API_KEY to run it)")

    passed = _check_gates(metrics)
    print("\n" + ("✅ ALL GATES PASSED" if passed else "❌ GATE FAILURE"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
