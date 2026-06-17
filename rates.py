"""Daily gold/silver rates ("bhav") + Sarafa-style price maths.

Rates in India come from the local **Sarafa Bazar** (the Sarafa Traders
Committee), which publishes a per-gram, per-karat sheet every day. There is no
free official API, so the jeweller enters that day's Sarafa rates (in the app
sidebar, or by editing rates.json) and Aurum uses them precisely.

PRICING METHOD (matches how Sarafa jewellers bill):
    per-gram price = karat rate + (karat rate × making% / 100)
    subtotal       = per-gram price × weight (g)
    total          = subtotal + 3% GST on the subtotal
Use the rate for the EXACT karat from the sheet — never derive 22K from 24K.
"""
from __future__ import annotations

import json

import config

DEFAULT_RATES = {
    "updated": "—",
    "city": "Jaipur",
    "source": "Sarafa Traders Committee",
    "currency": "INR",
    "per_gram": {"gold_24k": 15350, "gold_22k": 14230, "gold_18k": 11970,
                 "gold_14k": 9520, "silver": 253.5},
    "return_less_per_gram": 500,
    "note": "fallback Sarafa sample rates",
}

GST_PCT = 3.0   # Sarafa billing applies 3% GST on (metal value + making charges)


def load_rates() -> dict:
    if config.RATES_PATH.exists():
        try:
            return json.loads(config.RATES_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return DEFAULT_RATES


def save_rates(rates: dict) -> None:
    config.RATES_PATH.write_text(json.dumps(rates, indent=2), encoding="utf-8")


def _karat_rate(pg: dict, karat: int) -> float:
    """The directly-quoted Sarafa rate for a karat (no deriving from 24K)."""
    return {24: pg["gold_24k"], 22: pg["gold_22k"], 18: pg["gold_18k"],
            14: pg.get("gold_14k", pg["gold_24k"] * 14 / 24)}.get(karat)


def rates_block(rates: dict | None = None) -> str:
    """Text summary of today's Sarafa rates, injected into the LLM prompt."""
    rates = rates or load_rates()
    pg = rates["per_gram"]
    cur = rates.get("currency", "INR")
    ret = rates.get("return_less_per_gram")
    lines = [
        f"CURRENT GOLD & SILVER RATES (per gram, {cur}, "
        f"source: {rates.get('source','manual')}, "
        f"{rates.get('city','India')}, set on {rates.get('updated','—')}):",
        f"- 24K gold (approx): {pg['gold_24k']}/g",
        f"- 22K gold: {pg['gold_22k']}/g",
        f"- 18K gold: {pg['gold_18k']}/g",
    ]
    if "gold_14k" in pg:
        lines.append(f"- 14K gold: {pg['gold_14k']}/g")
    lines.append(f"- Silver: {pg['silver']}/g")
    if ret:
        lines.append(f"Buyback/return: {ret}/g LESS than the buying rate (22K/18K/14K).")
    lines.append("Quoted per gram per karat — use the exact karat rate. Rates change daily.")
    return "\n".join(lines)


def gold_rate(karat: int, rates: dict | None = None) -> float:
    return _karat_rate((rates or load_rates())["per_gram"], karat)


def silver_rate(rates: dict | None = None) -> float:
    return (rates or load_rates())["per_gram"]["silver"]


def price_breakdown(
    rate: float,
    weight_g: float,
    making_pct: float = 0.0,
    making_flat_per_g: float = 0.0,
    gst_pct: float = GST_PCT,
) -> dict:
    """Metal-agnostic Sarafa price: rate -> per-gram(+making) -> subtotal -> +3% GST."""
    metal_value = rate * weight_g
    making = (rate * making_pct / 100 + making_flat_per_g) * weight_g
    subtotal = metal_value + making
    gst = subtotal * gst_pct / 100
    return {
        "rate": rate,
        "per_gram_with_making": round(rate + rate * making_pct / 100 + making_flat_per_g, 2),
        "metal_value": round(metal_value, 2),
        "making": round(making, 2),
        "subtotal": round(subtotal, 2),
        "gst": round(gst, 2),
        "total": round(subtotal + gst, 2),
    }


def estimate_price(
    weight_g: float,
    karat: int = 22,
    making_pct: float = 0.0,
    making_flat_per_g: float = 0.0,
    rates: dict | None = None,
    gst_pct: float = GST_PCT,
) -> dict:
    """Itemised Sarafa-style price for a gold article (uses the quoted karat rate)."""
    rate = _karat_rate((rates or load_rates())["per_gram"], karat)
    return price_breakdown(rate, weight_g, making_pct, making_flat_per_g, gst_pct)


def estimate_silver(weight_g: float, making_pct: float = 0.0,
                    rates: dict | None = None) -> dict:
    """Itemised Sarafa-style price for a silver article."""
    return price_breakdown(silver_rate(rates), weight_g, making_pct)


if __name__ == "__main__":
    print(rates_block())
    print("\nExample — 12g 22K, 11% making (your Sarafa example):")
    print(estimate_price(12, karat=22, making_pct=11))
