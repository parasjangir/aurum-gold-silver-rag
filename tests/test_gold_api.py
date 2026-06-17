"""Tests for the GoldAPI.io live-rate mapping (mocked — no network)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

import gold_api

_FAKE_GOLD = {
    "price_gram_24k": 13130.97, "price_gram_22k": 12036.72,
    "price_gram_18k": 9848.22, "price_gram_14k": 7659.73,
    "timestamp": 1781682200, "currency": "INR",
}
_FAKE_SILVER = {"price_gram_24k": 212.63}


def test_fetch_live_rates_maps_fields(monkeypatch):
    monkeypatch.setattr(
        gold_api, "_fetch",
        lambda symbol, currency, key: _FAKE_GOLD if symbol == "XAU" else _FAKE_SILVER,
    )
    r = gold_api.fetch_live_rates(key="dummy")
    pg = r["per_gram"]
    assert pg["gold_24k"] == 13130.97
    assert pg["gold_22k"] == 12036.72
    assert pg["gold_18k"] == 9848.22
    assert pg["gold_14k"] == 7659.73
    assert pg["silver"] == 212.63
    assert r["source"] == "GoldAPI.io"
    assert r["currency"] == "INR"


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GOLDAPI_KEY", raising=False)
    with pytest.raises(gold_api.GoldAPIError):
        gold_api.fetch_live_rates(key=None)
