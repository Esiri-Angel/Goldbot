"""
gold_pattern_live.py

*** FOR TESTING FIRST -- not wired to Telegram or GitHub Actions yet. ***
Run this manually whenever you want a read on gold, same as
model_b_live.py, but using the pattern-scan approach instead of Model B's
daily/4h/ADX logic -- the one with an actual validated real-data result
(+0.193R on double bottom vs Model B's -0.060R, both on XAU/USD).

WHAT THIS CHECKS
  The exact 3 patterns validated in the pattern scan, each on the
  timeframe(s) where it actually tested positive:
    - Double bottom (buy)     -- 1h, 30m, 5m   [PRIMARY signal]
    - Head & shoulders (sell) -- 15m, 30m      [PRIMARY signal]
    - Bearish engulfing (sell)-- 15m, 5m       [CONFIRMATION ONLY --
                                                 not traded standalone,
                                                 per the document's own
                                                 guidance: weakest edge
                                                 of the three]

CONFIDENCE TIERS (matches the trading guidance already given)
  1h  = highest confidence -- best per-trade numbers, least noise
  30m = medium confidence -- earlier entries on the same setups
  15m/5m = lower confidence -- real signal, but noisier, size down

STACKED CONFIRMATION
  If bearish engulfing fires in the SAME direction as double bottom or
  head & shoulders within a few bars of each other, that's flagged as a
  higher-conviction setup worth noting -- exactly the "confirmation
  only" use case described for that pattern.

Same fixed SL/TP framework as the original scan (1.5x ATR stop, 2:1
reward:risk target) so live signals are measured the same way the
backtest that validated them was measured -- not a new, untested
mechanic bolted on for the live version.

RUN:
    python gold_pattern_live.py
"""

import os
from datetime import datetime, timezone

import pandas as pd
import numpy as np

from structure import analyze
from risk_management import compute_atr
from candlestick_patterns import bearish_engulfing
from chart_patterns import double_bottom, head_and_shoulders
from data_feed import get_candles

SYMBOL = "XAUUSD"
SL_ATR_MULT = 1.5
REWARD_RISK = 2.0
LOOKBACK_BARS = 3  # how many recent candles count as "still fresh"

# (timeframe, pattern_fn, direction, tier, standalone)
CHECKS = [
    ("1h", double_bottom, "long", "PRIMARY (1h)", True),
    ("30m", double_bottom, "long", "PRIMARY (30m)", True),
    ("5m", double_bottom, "long", "lower-confidence (5m)", True),
    ("15m", head_and_shoulders, "short", "PRIMARY (15m)", True),
    ("30m", head_and_shoulders, "short", "PRIMARY (30m)", True),
    ("15m", bearish_engulfing, "short", "confirmation-only (15m)", False),
    ("5m", bearish_engulfing, "short", "confirmation-only (5m)", False),
]

TIMEFRAMES_NEEDED = ["1h", "30m", "15m", "5m"]


def fetch_all_timeframes(symbol: str = SYMBOL, source: str = "yfinance", api_key: str = "") -> dict:
    lookback_map = {"1h": 1400, "30m": 2800, "15m": 5000, "5m": 15000}
    data = {}
    for tf in TIMEFRAMES_NEEDED:
        data[tf] = analyze(get_candles(symbol, interval=tf, lookback=lookback_map[tf],
                                        source=source, api_key=api_key))
    return data


def check_one(analyzed: pd.DataFrame, pattern_fn, direction: str, tier: str, standalone: bool) -> dict:
    """Checks whether a pattern fired within the last LOOKBACK_BARS candles."""
    try:
        sig = pattern_fn(analyzed)
    except Exception as exc:
        return {"fired": False, "error": str(exc)}

    recent = sig.tail(LOOKBACK_BARS)
    if not recent.any():
        return {"fired": False}

    fire_idx = recent[recent].index[-1]  # most recent True
    bars_ago = len(analyzed) - 1 - analyzed.index.get_loc(fire_idx)

    price = float(analyzed["close"].iloc[-1])
    atr = compute_atr(analyzed).iloc[-1]
    if pd.isna(atr) or atr <= 0:
        return {"fired": True, "bars_ago": bars_ago, "error": "ATR unavailable"}

    risk = atr * SL_ATR_MULT
    if direction == "long":
        sl, tp = price - risk, price + risk * REWARD_RISK
    else:
        sl, tp = price + risk, price - risk * REWARD_RISK

    return {
        "fired": True, "direction": direction, "tier": tier, "standalone": standalone,
        "bars_ago": bars_ago, "fire_time": fire_idx,
        "entry": round(price, 5), "sl": round(float(sl), 5), "tp": round(float(tp), 5),
        "reward_risk": REWARD_RISK,
    }


def check_gold_live(source: str = "yfinance", api_key: str = "", fetch_fn=None) -> dict:
    fetch_fn = fetch_fn or (lambda: fetch_all_timeframes(SYMBOL, source, api_key))
    data = fetch_fn()

    detections = []
    for tf, pattern_fn, direction, tier, standalone in CHECKS:
        result = check_one(data[tf], pattern_fn, direction, tier, standalone)
        if result.get("fired"):
            result["timeframe"] = tf
            result["pattern"] = pattern_fn.__name__.replace("_", " ")
            detections.append(result)

    # Stacked confirmation: bearish engulfing firing alongside a standalone
    # short signal (head & shoulders) within a few bars, same direction
    standalone_signals = [d for d in detections if d.get("standalone") and "error" not in d]
    confirmations = [d for d in detections if not d.get("standalone") and "error" not in d]
    stacked = []
    for s in standalone_signals:
        for c in confirmations:
            if s["direction"] == c["direction"] and abs(s["bars_ago"] - c["bars_ago"]) <= LOOKBACK_BARS:
                stacked.append((s, c))

    return {
        "checked_at": datetime.now(timezone.utc),
        "detections": detections,
        "standalone_signals": standalone_signals,
        "confirmations": confirmations,
        "stacked": stacked,
    }


def format_gold_report(result: dict) -> str:
    lines = ["=" * 74, f"GOLD PATTERN BOT -- {result['checked_at']:%Y-%m-%d %H:%M UTC}",
              "TESTING ONLY -- not yet wired to Telegram or scheduled alerts",
              "=" * 74, ""]

    standalone = result["standalone_signals"]
    confirmations = result["confirmations"]
    stacked = result["stacked"]

    if not standalone and not confirmations:
        lines.append("🔴 NO TRADE")
        lines.append("  Nothing fired on any timeframe in the last few candles.")
        lines.append("  A quiet scan is a normal, correct result -- these are real,")
        lines.append("  validated patterns, not something that fires constantly.")
        lines.append("")
        return "\n".join(lines)

    if standalone:
        lines.append(f"🟢 TRADE SETUP READY  ({len(standalone)})")
        for s in sorted(standalone, key=lambda d: d["bars_ago"]):
            lines.append(f"  {s['pattern'].title()}  --  {s['direction'].upper()}  --  {s['tier']}")
            lines.append(f"    Entry: {s['entry']}   SL: {s['sl']}   TP: {s['tp']}   (2:1)")
            lines.append(f"    Signal fired {s['bars_ago']} candle(s) ago on {s['timeframe']}")
        lines.append("")

    if stacked:
        lines.append(f"⭐ HIGHER CONVICTION  ({len(stacked)}) -- confirmation stacked with a primary signal")
        for s, c in stacked:
            lines.append(f"  {s['pattern'].title()} ({s['timeframe']}) confirmed by "
                         f"{c['pattern'].title()} ({c['timeframe']}), both {s['direction'].upper()}")
        lines.append("")

    if confirmations and not stacked:
        lines.append(f"🟡 CONFIRMATION-ONLY SIGNALS  ({len(confirmations)}) -- not a standalone trade")
        for c in confirmations:
            lines.append(f"  {c['pattern'].title()}  --  {c['direction'].upper()}  --  {c['tier']}")
            lines.append(f"    (bearish engulfing alone is the weakest edge of the three -- wait for")
            lines.append(f"     it to line up with double bottom or head & shoulders before acting on it)")
        lines.append("")

    lines.append("-" * 74)
    lines.append("SL/TP use 1.5x ATR stop, 2:1 reward:risk -- the same rules the backtest")
    lines.append("was measured with. This is a smaller, newer validated result (47 trades")
    lines.append("on double bottom's best timeframe) than a fully mature system -- treat")
    lines.append("early signals as data to log, not a proven track record yet.")
    return "\n".join(lines)


if __name__ == "__main__":
    result = check_gold_live()
    report = format_gold_report(result)
    print(report)

    os.makedirs("reports", exist_ok=True)
    path = f"reports/gold-pattern-check-{datetime.now(timezone.utc):%Y%m%d-%H%M}.txt"
    with open(path, "w") as f:
        f.write(report)
    print(f"\nSaved to {path}")
