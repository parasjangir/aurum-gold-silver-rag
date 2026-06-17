"""Tests for the Sarafa-style rate & price maths."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from rates import estimate_price, load_rates, rates_block

# The exact Sarafa example the user gave us.
SARAFA = {
    "per_gram": {"gold_24k": 15350, "gold_22k": 14230, "gold_18k": 11970,
                 "gold_14k": 9520, "silver": 253.5}
}


def test_sarafa_price_example_is_exact():
    # 12 g, 22K @ 14230/g, 11% making, 3% GST on subtotal -> 195229.91
    r = estimate_price(12, karat=22, making_pct=11, rates=SARAFA)
    assert r["per_gram_with_making"] == pytest.approx(15795.30, abs=0.01)
    assert r["subtotal"] == pytest.approx(189543.60, abs=0.01)
    assert r["total"] == pytest.approx(195229.91, abs=0.02)


def test_gst_is_three_percent_of_subtotal():
    r = estimate_price(10, karat=22, making_pct=0, rates=SARAFA)
    # no making -> subtotal is pure metal; GST is 3% of it
    assert r["gst"] == pytest.approx(r["subtotal"] * 0.03, abs=0.01)


def test_uses_quoted_karat_rate_not_derived():
    # 18K must use the quoted 11970, not 24K x 18/24 (= 11512.5)
    r = estimate_price(1, karat=18, making_pct=0, rates=SARAFA)
    assert r["metal_value"] == pytest.approx(11970, abs=0.01)


def test_rates_block_lists_karats_and_source():
    block = rates_block(load_rates())
    assert "22K" in block and "14K" in block
    assert "source" in block.lower()
