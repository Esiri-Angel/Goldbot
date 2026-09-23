"""
structure.py
ICT-style market structure detection: swing points, break of structure (BOS),
change of character (CHoCH), order blocks, and fair value gaps (FVGs).

All functions take a DataFrame with columns: open, high, low, close
(indexed by time) and return the same DataFrame with extra columns, or
a separate list of detected events.
"""

import pandas as pd
import numpy as np


def find_swing_points(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    """
    A candle is a swing HIGH if its high is the max within [i-left, i+right].
    A candle is a swing LOW if its low is the min within [i-left, i+right].
    Adds boolean columns 'swing_high' and 'swing_low'.
    """
    df = df.copy()
    n = len(df)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)

    highs = df["high"].values
    lows = df["low"].values

    for i in range(left, n - right):
        window_high = highs[i - left: i + right + 1]
        window_low = lows[i - left: i + right + 1]
        if highs[i] == window_high.max() and np.argmax(window_high) == left:
            swing_high[i] = True
        if lows[i] == window_low.min() and np.argmin(window_low) == left:
            swing_low[i] = True

    df["swing_high"] = swing_high
    df["swing_low"] = swing_low
    return df


def detect_structure_breaks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Walks through swing points to classify trend and flag:
      - BOS (break of structure): continuation - price breaks the most recent
        swing in the direction of the existing trend
      - CHoCH (change of character): reversal - price breaks a swing against
        the prevailing trend, signalling a possible trend change

    Adds a 'structure_event' column: None, 'BOS_up', 'BOS_down', 'CHoCH_up', 'CHoCH_down'
    Adds a 'trend' column: 'up', 'down', or None (unknown, not enough data yet)
    """
    df = df.copy()
    events = [None] * len(df)
    trend_col = [None] * len(df)

    last_swing_high = None
    last_swing_low = None
    trend = None  # 'up' or 'down'

    for i in range(len(df)):
        row = df.iloc[i]

        # Update trend based on breaking prior swing high/low
        if last_swing_high is not None and row["close"] > last_swing_high:
            if trend == "down":
                events[i] = "CHoCH_up"
            elif trend == "up":
                events[i] = "BOS_up"
            else:
                events[i] = "BOS_up"
            trend = "up"
            last_swing_high = None  # consumed; wait for the next one

        if last_swing_low is not None and row["close"] < last_swing_low:
            if trend == "up":
                events[i] = "CHoCH_down"
            elif trend == "down":
                events[i] = "BOS_down"
            else:
                events[i] = "BOS_down"
            trend = "down"
            last_swing_low = None

        # Track the most recent swing high/low for future comparisons
        if row["swing_high"]:
            last_swing_high = row["high"]
        if row["swing_low"]:
            last_swing_low = row["low"]

        trend_col[i] = trend

    df["structure_event"] = events
    df["trend"] = trend_col
    return df


def detect_order_blocks(df: pd.DataFrame, lookback: int = 1) -> pd.DataFrame:
    """
    Simplified order block detection: the last opposite-colored candle
    immediately before a strong impulsive move (a BOS/CHoCH event) is
    marked as an order block zone.

    Adds 'order_block' column: None, 'bullish_ob', 'bearish_ob'
    """
    df = df.copy()
    ob = [None] * len(df)

    for i in range(lookback, len(df)):
        event = df["structure_event"].iloc[i]
        if event in ("BOS_up", "CHoCH_up"):
            # look back for the last bearish (down) candle before the up move
            for j in range(i - 1, max(i - 6, -1), -1):
                if df["close"].iloc[j] < df["open"].iloc[j]:
                    ob[j] = "bullish_ob"
                    break
        elif event in ("BOS_down", "CHoCH_down"):
            for j in range(i - 1, max(i - 6, -1), -1):
                if df["close"].iloc[j] > df["open"].iloc[j]:
                    ob[j] = "bearish_ob"
                    break

    df["order_block"] = ob
    return df


def detect_fair_value_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    A bullish FVG: candle i-1's high < candle i+1's low (gap left behind by an
    impulsive move up). Bearish FVG is the mirror case.

    Adds 'fvg' column: None, 'bullish_fvg', 'bearish_fvg'
    Adds 'fvg_top' / 'fvg_bottom' for the gap boundaries (NaN if none)
    """
    df = df.copy()
    n = len(df)
    fvg = [None] * n
    fvg_top = [np.nan] * n
    fvg_bottom = [np.nan] * n

    highs = df["high"].values
    lows = df["low"].values

    for i in range(1, n - 1):
        if highs[i - 1] < lows[i + 1]:
            fvg[i] = "bullish_fvg"
            fvg_bottom[i] = highs[i - 1]
            fvg_top[i] = lows[i + 1]
        elif lows[i - 1] > highs[i + 1]:
            fvg[i] = "bearish_fvg"
            fvg_top[i] = lows[i - 1]
            fvg_bottom[i] = highs[i + 1]

    df["fvg"] = fvg
    df["fvg_top"] = fvg_top
    df["fvg_bottom"] = fvg_bottom
    return df


def analyze(df: pd.DataFrame, swing_left: int = 3, swing_right: int = 3) -> pd.DataFrame:
    """Run the full structure pipeline and return an annotated DataFrame."""
    df = find_swing_points(df, left=swing_left, right=swing_right)
    df = detect_structure_breaks(df)
    df = detect_order_blocks(df)
    df = detect_fair_value_gaps(df)
    return df


def summarize_latest(df: pd.DataFrame, symbol: str = "") -> str:
    """Produce a short human-readable summary of the current structural state."""
    latest = df.iloc[-1]
    trend = latest["trend"] or "undetermined"

    recent_events = df[df["structure_event"].notna()].tail(3)
    event_lines = [
        f"  {idx}: {row['structure_event']} @ {row['close']:.4f}"
        for idx, row in recent_events.iterrows()
    ]

    open_obs = df[df["order_block"].notna()].tail(3)
    ob_lines = [
        f"  {idx}: {row['order_block']} (range {row['low']:.4f}-{row['high']:.4f})"
        for idx, row in open_obs.iterrows()
    ]

    open_fvgs = df[df["fvg"].notna()].tail(3)
    fvg_lines = [
        f"  {idx}: {row['fvg']} ({row['fvg_bottom']:.4f}-{row['fvg_top']:.4f})"
        for idx, row in open_fvgs.iterrows()
    ]

    lines = [
        f"=== Structure summary for {symbol or 'instrument'} ===",
        f"Current trend: {trend}",
        f"Last close: {latest['close']:.4f}",
        "",
        "Recent structure events:",
        *(event_lines or ["  (none detected yet)"]),
        "",
        "Recent order blocks:",
        *(ob_lines or ["  (none detected yet)"]),
        "",
        "Recent fair value gaps:",
        *(fvg_lines or ["  (none detected yet)"]),
    ]
    return "\n".join(lines)
