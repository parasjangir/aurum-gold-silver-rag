"""Live gold/silver rates from GoldAPI.io (https://www.goldapi.io).

GoldAPI returns live SPOT prices per gram for each karat in a given currency.
We map XAU (gold) + XAG (silver) for INR into our rates dict and cache it to
rates.json — so the rest of the app is unchanged: it still reads rates.json,
which is now kept fresh from the API instead of typed by hand.

NOTE: these are international spot rates. A local Sarafa committee rate adds
local duty/premium and will run higher; tune `return_less_per_gram` (the buyback
spread) or add a premium if you want it to match a specific shop exactly.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import date, datetime

from dotenv import load_dotenv

import config
from rates import save_rates

load_dotenv(config.PROJECT_ROOT / ".env")
_BASE = "https://www.goldapi.io/api"


class GoldAPIError(RuntimeError):
    """Raised when the live rate fetch fails (missing key, network, HTTP error)."""


def _fetch(symbol: str, currency: str, key: str) -> dict:
    req = urllib.request.Request(
        f"{_BASE}/{symbol}/{currency}",
        headers={"x-access-token": key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise GoldAPIError(f"GoldAPI HTTP {e.code} for {symbol}/{currency}: {e.reason}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise GoldAPIError(f"GoldAPI network error: {e}") from e


def fetch_live_rates(key: str | None = None, currency: str = config.RATES_CURRENCY) -> dict:
    """Fetch live gold + silver rates and return them in our rates-dict shape."""
    key = key or os.getenv("GOLDAPI_KEY")
    if not key:
        raise GoldAPIError(
            "GOLDAPI_KEY not set. Add it to sona-rag/.env: GOLDAPI_KEY=goldapi-...-io"
        )
    gold = _fetch("XAU", currency, key)
    silver = _fetch("XAG", currency, key)

    ts = gold.get("timestamp")
    when = (
        datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        if ts else date.today().isoformat()
    )
    return {
        "updated": when,
        "city": "GoldAPI.io · live spot",
        "source": "GoldAPI.io",
        "currency": currency,
        "per_gram": {
            "gold_24k": round(gold["price_gram_24k"], 2),
            "gold_22k": round(gold["price_gram_22k"], 2),
            "gold_18k": round(gold["price_gram_18k"], 2),
            "gold_14k": round(gold["price_gram_14k"], 2),
            "silver": round(silver["price_gram_24k"], 2),
        },
        "return_less_per_gram": 500,
        "note": "Live spot from GoldAPI.io (XAU/XAG, INR). Buyback spread is a local convention.",
    }


def refresh_rates(key: str | None = None) -> dict:
    """Fetch live rates and write them to rates.json (the app's source of truth)."""
    rates = fetch_live_rates(key)
    save_rates(rates)
    return rates


if __name__ == "__main__":
    import rates as r
    print("Fetching live rates from GoldAPI.io…\n")
    print(r.rates_block(refresh_rates()))
