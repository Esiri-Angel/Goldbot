"""
chart_patterns.py
Detects classic chart patterns using swing highs/lows (reuses
structure.py's fractal swing detection, already validated elsewhere in
this project). These are heuristic by nature -- chart patterns are fuzzy
concepts even to human traders -- so the rules here are kept simple,
fixed, and documented rather than tuned to make any backtest look good.

Patterns included:
  - Double Top / Double Bottom
  - Head & Shoulders / Inverse Head & Shoulders

NOT included: triangles, flags, wedges -- these require trendline slope
fitting that is much more prone to false positives / curve-fitting than
the patterns above. Can be added later if the simpler patterns show a
real, validated edge worth building on.
"""

import pandas as pd
import numpy as np

from structure import analyze


def _swing_highs(df):
    return df[df["swing_high"]][["high"]].rename(columns={"high": "price"})


def _swing_lows(df):
    return df[df["swing_low"]][["low"]].rename(columns={"low": "price"})


def double_top(df, tolerance=0.015, min_gap=3):
    """
    Two swing highs within `tolerance` (fractional) of each other, with at
    least one swing low between them (the 'neckline'). Signal fires on the
    candle where price closes below that neckline low after the second top.
    """
    analyzed = df if "swing_high" in df.columns else analyze(df)
    highs = analyzed[analyzed["swing_high"]]
    lows = analyzed[analyzed["swing_low"]]
    signal = pd.Series(False, index=analyzed.index)

    high_idx = list(highs.index)
    for i in range(1, len(high_idx)):
        t1, t2 = high_idx[i - 1], high_idx[i]
        p1, p2 = highs.loc[t1, "high"], highs.loc[t2, "high"]
        if abs(p1 - p2) / p1 > tolerance:
            continue
        between_lows = lows[(lows.index > t1) & (lows.index < t2)]
        if between_lows.empty:
            continue
        neckline = between_lows["low"].min()
        after = analyzed[analyzed.index > t2]
        breaks = after[after["close"] < neckline]
        if not breaks.empty:
            signal.loc[breaks.index[0]] = True
    return signal


def double_bottom(df, tolerance=0.015, min_gap=3):
    """Mirror of double_top: two swing lows near the same level, signal on breakout above the neckline high."""
    analyzed = df if "swing_high" in df.columns else analyze(df)
    highs = analyzed[analyzed["swing_high"]]
    lows = analyzed[analyzed["swing_low"]]
    signal = pd.Series(False, index=analyzed.index)

    low_idx = list(lows.index)
    for i in range(1, len(low_idx)):
        t1, t2 = low_idx[i - 1], low_idx[i]
        p1, p2 = lows.loc[t1, "low"], lows.loc[t2, "low"]
        if abs(p1 - p2) / p1 > tolerance:
            continue
        between_highs = highs[(highs.index > t1) & (highs.index < t2)]
        if between_highs.empty:
            continue
        neckline = between_highs["high"].max()
        after = analyzed[analyzed.index > t2]
        breaks = after[after["close"] > neckline]
        if not breaks.empty:
            signal.loc[breaks.index[0]] = True
    return signal


def head_and_shoulders(df, shoulder_tolerance=0.02):
    """Three consecutive swing highs: left shoulder, head (highest), right shoulder (~= left).
    Signal fires when price closes below the neckline (lower of the two intervening swing lows)."""
    analyzed = df if "swing_high" in df.columns else analyze(df)
    highs = analyzed[analyzed["swing_high"]]
    lows = analyzed[analyzed["swing_low"]]
    signal = pd.Series(False, index=analyzed.index)

    high_idx = list(highs.index)
    for i in range(2, len(high_idx)):
        t_l, t_h, t_r = high_idx[i - 2], high_idx[i - 1], high_idx[i]
        p_l, p_h, p_r = highs.loc[t_l, "high"], highs.loc[t_h, "high"], highs.loc[t_r, "high"]
        if not (p_h > p_l and p_h > p_r):
            continue
        if abs(p_l - p_r) / p_l > shoulder_tolerance:
            continue
        between_lows = lows[(lows.index > t_l) & (lows.index < t_r)]
        if len(between_lows) < 2:
            continue
        neckline = between_lows["low"].min()
        after = analyzed[analyzed.index > t_r]
        breaks = after[after["close"] < neckline]
        if not breaks.empty:
            signal.loc[breaks.index[0]] = True
    return signal


def inverse_head_and_shoulders(df, shoulder_tolerance=0.02):
    """Mirror of head_and_shoulders using swing lows."""
    analyzed = df if "swing_high" in df.columns else analyze(df)
    highs = analyzed[analyzed["swing_high"]]
    lows = analyzed[analyzed["swing_low"]]
    signal = pd.Series(False, index=analyzed.index)

    low_idx = list(lows.index)
    for i in range(2, len(low_idx)):
        t_l, t_h, t_r = low_idx[i - 2], low_idx[i - 1], low_idx[i]
        p_l, p_h, p_r = lows.loc[t_l, "low"], lows.loc[t_h, "low"], lows.loc[t_r, "low"]
        if not (p_h < p_l and p_h < p_r):
            continue
        if abs(p_l - p_r) / p_l > shoulder_tolerance:
            continue
        between_highs = highs[(highs.index > t_l) & (highs.index < t_r)]
        if len(between_highs) < 2:
            continue
        neckline = between_highs["high"].max()
        after = analyzed[analyzed.index > t_r]
        breaks = after[after["close"] > neckline]
        if not breaks.empty:
            signal.loc[breaks.index[0]] = True
    return signal


# Registry: name -> (detector_fn, implied_direction)
CHART_PATTERNS = {
    "double_top": (double_top, "short"),
    "double_bottom": (double_bottom, "long"),
    "head_and_shoulders": (head_and_shoulders, "short"),
    "inverse_head_and_shoulders": (inverse_head_and_shoulders, "long"),
}
