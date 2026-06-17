"""The RAG pipeline: retrieve -> ground -> generate -> cite.

Aurum is a gold/silver EXPERT, not a strict refuser. Earlier versions said
"I don't know" whenever the answer wasn't word-for-word in the docs — too narrow
for a real jeweller's assistant. Now:

  * Off-domain questions (weather, cricket) are refused cheaply, before any LLM
    call, using a low retrieval-similarity floor.
  * In-domain questions are always answered, using three sources of truth:
      1. CURRENT RATES   — today's bhav, injected every turn (for rate/price Qs).
      2. REFERENCE PASSAGES — retrieved KB chunks, cited as [n] when used.
      3. The model's own gold/silver expertise — allowed as a fallback, but it
         must flag general/estimated info and must NOT invent specific live
         prices or exact legal figures.

This keeps answers broad AND grounded: India-specific facts come from the KB,
live numbers come from the rates, and the model fills sensible gaps honestly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import config
import llm
import vector_store as vs
from rates import estimate_price, estimate_silver, load_rates, rates_block

OFF_DOMAIN_MESSAGE = (
    f"I'm {config.APP_NAME} — I focus on gold, silver, jewellery, hallmarking, "
    "rates, and pricing. Ask me anything in that world and I've got you."
)

SYSTEM_PROMPT = (
    f"You are {config.APP_NAME}, an expert assistant on GOLD and SILVER in India — "
    "purity and karat, BIS hallmarking, making charges, GST, gold/silver rates "
    "(bhav), pricing, and buying/selling/investment. You speak like a trusted, "
    "practical jeweller: clear, accurate, and helpful.\n\n"
    "How to answer:\n"
    "- Treat the REFERENCE PASSAGES as authoritative for India-specific facts, "
    "rules, and definitions. Every sentence that uses a passage MUST end with its "
    'number in brackets, e.g. "22K gold is 91.6% pure [1]."\n'
    "- The CURRENT RATES block holds today's LIVE per-karat rates (its source and "
    "timestamp are shown). For any rate/bhav or pricing question, use those exact "
    "rates. State the timestamp and that rates change continuously.\n"
    "- PRICING METHOD — always use this EXACT method (it's how Sarafa jewellers bill):\n"
    "    1) per-gram price = karat rate + (karat rate × making% ÷ 100)\n"
    "    2) subtotal       = per-gram price × weight in grams\n"
    "    3) total          = subtotal + 3% GST on the subtotal\n"
    "  Use the directly-quoted rate for the EXACT karat (never derive 22K from 24K).\n"
    "  Worked example: 12 g of 22K at 14230/g with 11% making → per-gram = "
    "14230 + 14230×0.11 = 15795.3; subtotal = 15795.3 × 12 = 189543.6; "
    "total = 189543.6 × 1.03 = ₹195229.91. Show your steps like this.\n"
    "- If a PRICE BREAKDOWN is provided below, those figures are AUTHORITATIVE. "
    "Present ONLY those exact numbers in clean prose/bullets. Do NOT show your own "
    "arithmetic or any alternative numbers — just give the breakdown and total "
    "naturally, as if you calculated it.\n"
    "- If the passages don't cover something, you MAY use your general gold/silver "
    "expertise, but clearly flag it as general guidance and NEVER fabricate "
    "specific current prices, exact tax percentages, or legal specifics.\n"
    "- If the question is NOT about gold, silver, jewellery, or precious metals, "
    "politely decline in one short sentence and invite a gold/silver question.\n"
    "- Be concise and well-structured. Use short paragraphs or bullets."
)

PROMPT_TEMPLATE = (
    "{rates}\n\n"
    "{computed}"
    "REFERENCE PASSAGES:\n{context}\n\n"
    "USER QUESTION: {question}\n\n"
    "Answer as {app}. Cite reference passages you use like [1] (never cite the "
    "rates or price-breakdown blocks). If a PRICE BREAKDOWN is present, present "
    "exactly those numbers as the breakdown and total — show no other arithmetic."
)


def _parse_price_query(question: str) -> dict | None:
    """Pull weight / karat / making% out of a pricing question, if present.

    We compute the price in Python (exact) rather than trust the LLM's mental
    arithmetic, which drifts. Returns None when it isn't a concrete price query.
    """
    q = question.lower()
    # Weight — English units OR Devanagari ग्राम/ग्रा (Hindi/Marwari).
    wm = re.search(r"(\d+(?:\.\d+)?)\s*(?:grams?|gms?|g|ग्राम|ग्रा|जीएम)(?![a-z])", q)
    if not wm:
        return None
    weight = float(wm.group(1))
    mm = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent|pct|प्रतिशत|फीसदी)", q)
    making = float(mm.group(1)) if mm else 0.0
    km = re.search(r"(\d+)\s*(?:karat|carat|kt|k|कैरेट|कैरट|केरेट)(?![a-z])", q)
    silver_words = ("silver", "चाँदी", "चांदी", "चान्दी")
    gold_words = ("gold", "सोना", "सोने", "सोनो", "स्वर्ण")
    if km:
        return {"metal": "gold", "weight": weight, "karat": int(km.group(1)), "making": making}
    if any(w in q for w in silver_words):
        return {"metal": "silver", "weight": weight, "karat": None, "making": making}
    if any(w in q for w in gold_words):  # gold + weight, no karat → assume 22K
        return {"metal": "gold", "weight": weight, "karat": 22, "making": making}
    return None


# Localised intro/note for the deterministic price breakdown (itemised labels
# stay in standard English terms, as on most Indian jeweller bills).
PRICE_TEXT = {
    "English": {"intro": "Here's the exact price breakdown:",
                "note": "Rates as of {when} · source {src} · rates change continuously."},
    "हिन्दी": {"intro": "कीमत का पूरा ब्यौरा:",
               "note": "भाव {when} तक · स्रोत {src} · भाव लगातार बदलते हैं।"},
    "Hinglish": {"intro": "Yeh raha exact price breakdown:",
                 "note": "Bhav {when} tak · source {src} · bhav lagataar badalte hain."},
    "मारवाड़ी": {"intro": "कीमत रो पूरो ब्यौरो:",
                "note": "भाव {when} तांई · स्रोत {src} · भाव लगातार बदलै।"},
}


def _render_price_answer(parsed: dict, language: str) -> str:
    """Build the price breakdown entirely in Python — exact, no LLM involved."""
    rates = load_rates()
    if parsed["metal"] == "silver":
        bd = estimate_silver(parsed["weight"], parsed["making"], rates)
        rate_label = "Silver rate"
    else:
        bd = estimate_price(parsed["weight"], karat=parsed["karat"],
                            making_pct=parsed["making"], rates=rates)
        rate_label = f"{parsed['karat']}K gold rate"

    txt = PRICE_TEXT.get(language, PRICE_TEXT["English"])
    return "\n".join([
        txt["intro"], "",
        f"- {rate_label}: ₹{bd['rate']:,.2f}/g",
        f"- Weight: {parsed['weight']:g} g",
        f"- Making: {parsed['making']:g}%",
        f"- Per-gram price (rate + making): ₹{bd['per_gram_with_making']:,.2f}",
        f"- Metal value: ₹{bd['metal_value']:,.2f}",
        f"- Making charges: ₹{bd['making']:,.2f}",
        f"- Subtotal: ₹{bd['subtotal']:,.2f}",
        f"- GST (3%): ₹{bd['gst']:,.2f}",
        f"- **TOTAL: ₹{bd['total']:,.2f}**", "",
        txt["note"].format(when=rates.get("updated", "—"), src=rates.get("source", "manual")),
    ])


# Supported answer languages. `instruction` tells the LLM how to write;
# `off_domain` is the localized refusal used by the deterministic guard.
LANGUAGES = {
    "English": {
        "instruction": "English",
        "off_domain": OFF_DOMAIN_MESSAGE,
    },
    "हिन्दी": {
        "instruction": "Hindi using the Devanagari script",
        "off_domain": "मैं Aurum हूँ — मैं सोना, चाँदी, ज़ेवर, हॉलमार्किंग, भाव और "
        "कीमत के बारे में बताता हूँ। इसी विषय में कुछ भी पूछिए! 🪙",
    },
    "Hinglish": {
        "instruction": "Hinglish — Hindi written in Roman/English letters, casual and friendly",
        "off_domain": "Main Aurum hoon — main sirf sona, chaandi, jewellery, "
        "hallmarking, bhav aur pricing ke baare mein batata hoon. Isi duniya ka "
        "kuch bhi poochho! 🪙",
    },
    "मारवाड़ी": {
        "instruction": "Marwari (Rajasthani) using the Devanagari script",
        "off_domain": "म्हैं Aurum हूँ — म्हैं सोनो, चाँदी, गहणा, हॉलमार्किंग अर "
        "भाव री बात करूँ। इणी बारै मांय कीं ई पूछो! 🪙",
    },
}


@dataclass
class Answer:
    question: str
    text: str
    found: bool                          # True if in-domain and answered
    sources: dict[int, str] = field(default_factory=dict)


def answer(question: str, language: str = "English") -> Answer:
    """Answer a gold/silver question in `language`; refuse off-domain ones."""
    lang = LANGUAGES.get(language, LANGUAGES["English"])

    # Price queries are answered DETERMINISTICALLY in Python — exact every time,
    # no LLM arithmetic, no tokens, and they work even if the LLM quota is gone.
    parsed = _parse_price_query(question)
    if parsed:
        return Answer(question, _render_price_answer(parsed, language), found=True)

    hits = vs.search(question, k=config.TOP_K)
    top_sim = hits[0]["similarity"] if hits else 0.0

    # Cheap, deterministic off-domain guard (no LLM call, no tokens).
    if top_sim < config.OFF_DOMAIN_FLOOR:
        return Answer(question, lang["off_domain"], found=False)

    # Passages strong enough to trust as cited grounding.
    kept = [h for h in hits if h["similarity"] >= config.GROUNDING_MIN_SIM]
    if kept:
        context = "\n\n".join(
            f"[{i}] (source: {h['source']})\n{h['text']}" for i, h in enumerate(kept, 1)
        )
    else:
        context = "(No specific passage strongly matched — rely on the current " \
                  "rates and your general gold/silver expertise, and flag estimates.)"

    system = (
        SYSTEM_PROMPT
        + f"\n- Write your ENTIRE answer in {lang['instruction']}. Keep numbers, "
        "₹ amounts, karat values, units, and [n] citations exactly as they are."
    )
    prompt = PROMPT_TEMPLATE.format(
        rates=rates_block(), computed="", context=context,
        question=question, app=config.APP_NAME,
    )
    text = llm.generate(prompt, system=system)

    # If the model still judged it off-domain, treat as a refusal.
    if text.strip() in (lang["off_domain"], OFF_DOMAIN_MESSAGE):
        return Answer(question, lang["off_domain"], found=False)

    sources = {i: h["source"] for i, h in enumerate(kept, 1)}
    return Answer(question, text, found=True, sources=sources)


def _format_sources(sources: dict[int, str]) -> str:
    if not sources:
        return ""
    return "Sources:\n" + "\n".join(f"  [{n}] {src}" for n, src in sources.items())


if __name__ == "__main__":
    print(f"{config.APP_NAME} — ask about gold, silver, rates, making charges, GST...")
    print("(type 'quit' to exit)\n")
    try:
        while True:
            q = input("You: ").strip()
            if q.lower() in {"quit", "exit", ""}:
                break
            r = answer(q)
            print(f"\n{config.APP_NAME}: {r.text}")
            footer = _format_sources(r.sources)
            if footer:
                print(footer)
            print()
    except (EOFError, KeyboardInterrupt):
        print("\nbye 👋")
    except llm.LLMError as e:
        print(f"\n[LLM not ready] {e}")
