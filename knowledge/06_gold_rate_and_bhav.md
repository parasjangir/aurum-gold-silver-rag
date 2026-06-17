# Gold Rate & Bhav — How Pricing Works

> Educational sample for Aurum. Live rates change daily; Aurum reads the current
> bhav from the rates the jeweller sets, not from this document.

## What "bhav" means

"Bhav" is the everyday word for the current market rate of gold or silver. It is
quoted **per gram** or **per 10 grams**, and sometimes per **tola**
(1 tola = 11.6638 g). The rate you hear on the news is usually for **24K** gold.

## Per-karat rates (Sarafa quotes each directly)

The Sarafa Traders Committee publishes a separate per-gram rate for each karat
(24K, 22K, 18K, 14K) and for silver, every day. Always use the directly-quoted
rate for the karat you're buying — for example 22K at ₹14,230/g — rather than
deriving the 22K rate from the 24K rate. (As a rough sanity check 22K ≈ 24K ×
22/24, but the committee's quoted per-karat figure is the one that counts.)

The committee sheet also lists a **return/buyback** rate — typically a fixed
amount per gram (e.g. ₹500/g) **less** than the buying rate, applied when you
sell 22K/18K/14K gold back.

## Why your shop's rate differs from the "news" rate

The headline rate is a benchmark. The price you actually pay at a shop also
includes making charges and GST, and the shop's quoted metal rate may differ
slightly from the benchmark. Two shops can show different "rates" because some
quote an all-inclusive figure while others quote the bare metal rate and add
making/GST separately.

## What moves the gold rate

- **International gold price** (USD per troy ounce) — the global anchor.
- **USD–INR exchange rate** — a weaker rupee raises the local price.
- **Import duty and taxes** in India.
- **Demand** — festivals and wedding season lift demand and price.
- **Interest rates and economic uncertainty** — gold is a safe-haven asset, so
  uncertainty tends to push prices up.

## Units cheat-sheet

- 1 tola = 11.6638 grams
- Rates are commonly quoted per gram or per 10 grams
- 1 troy ounce = 31.1035 grams (used for the international price)

## Computing what you'll pay

Final price for a gold ornament is built up the Sarafa way:

    per-gram price = karat rate + (karat rate × making%)
    subtotal       = per-gram price × net weight (g)
    final          = subtotal + 3% GST on the subtotal

Aurum can do this calculation for you using today's Sarafa bhav — just give it
the weight, the karat, and the making charge.
