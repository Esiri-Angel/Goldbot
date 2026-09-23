"""
candlestick_patterns.py
Detects classic candlestick patterns from OHLC data. Each detector returns
a boolean Series aligned to the dataframe's index, True on the candle
where the pattern completes.

Patterns included (bullish and bearish where applicable):
  - Bullish/Bearish Engulfing
  - Hammer / Shooting Star
  - Doji
  - Morning Star / Evening Star (3-candle)
  - Three White Soldiers / Three Black Crows

Definitions follow standard, widely-published rules (not tuned to any
backtest result) -- keeping them fixed and mechanical is what makes the
later win-rate scan meaningful rather than circular.
"""

import pandas as pd
import numpy as np


def _body(df):
    return (df["close"] - df["open"]).abs()


def _range(df):
    return df["high"] - df["low"]


def _upper_wick(df):
    return df["high"] - df[["open", "close"]].max(axis=1)


def _lower_wick(df):
    return df[["open", "close"]].min(axis=1) - df["low"]


def bullish_engulfing(df):
    prev_bear = df["close"].shift(1) < df["open"].shift(1)
    curr_bull = df["close"] > df["open"]
    engulfs = (df["open"] <= df["close"].shift(1)) & (df["close"] >= df["open"].shift(1))
    return prev_bear & curr_bull & engulfs


def bearish_engulfing(df):
    prev_bull = df["close"].shift(1) > df["open"].shift(1)
    curr_bear = df["close"] < df["open"]
    engulfs = (df["open"] >= df["close"].shift(1)) & (df["close"] <= df["open"].shift(1))
    return prev_bull & curr_bear & engulfs


def hammer(df):
    body = _body(df)
    rng = _range(df).replace(0, np.nan)
    lower = _lower_wick(df)
    upper = _upper_wick(df)
    small_body = body <= rng * 0.35
    long_lower = lower >= body * 2
    small_upper = upper <= body * 0.5
    return small_body & long_lower & small_upper & (rng > 0)


def shooting_star(df):
    body = _body(df)
    rng = _range(df).replace(0, np.nan)
    lower = _lower_wick(df)
    upper = _upper_wick(df)
    small_body = body <= rng * 0.35
    long_upper = upper >= body * 2
    small_lower = lower <= body * 0.5
    return small_body & long_upper & small_lower & (rng > 0)


def doji(df, threshold=0.1):
    body = _body(df)
    rng = _range(df).replace(0, np.nan)
    return (body <= rng * threshold) & (rng > 0)


def morning_star(df):
    c1_bear = df["close"].shift(2) < df["open"].shift(2)
    c1_body = _body(df).shift(2)
    c2_small = _body(df).shift(1) <= c1_body * 0.5
    c2_gap_down = df[["open", "close"]].max(axis=1).shift(1) < df["close"].shift(2)
    c3_bull = df["close"] > df["open"]
    c3_closes_into_c1 = df["close"] > (df["open"].shift(2) + df["close"].shift(2)) / 2
    return c1_bear & c2_small & c2_gap_down & c3_bull & c3_closes_into_c1


def evening_star(df):
    c1_bull = df["close"].shift(2) > df["open"].shift(2)
    c1_body = _body(df).shift(2)
    c2_small = _body(df).shift(1) <= c1_body * 0.5
    c2_gap_up = df[["open", "close"]].min(axis=1).shift(1) > df["close"].shift(2)
    c3_bear = df["close"] < df["open"]
    c3_closes_into_c1 = df["close"] < (df["open"].shift(2) + df["close"].shift(2)) / 2
    return c1_bull & c2_small & c2_gap_up & c3_bear & c3_closes_into_c1


def three_white_soldiers(df):
    bull = df["close"] > df["open"]
    all_bull = bull & bull.shift(1).fillna(False) & bull.shift(2).fillna(False)
    higher_closes = (df["close"] > df["close"].shift(1)) & (df["close"].shift(1) > df["close"].shift(2))
    opens_within_prev_body = (df["open"] > df["open"].shift(1)) & (df["open"] < df["close"].shift(1))
    return all_bull & higher_closes & opens_within_prev_body


def three_black_crows(df):
    bear = df["close"] < df["open"]
    all_bear = bear & bear.shift(1).fillna(False) & bear.shift(2).fillna(False)
    lower_closes = (df["close"] < df["close"].shift(1)) & (df["close"].shift(1) < df["close"].shift(2))
    opens_within_prev_body = (df["open"] < df["open"].shift(1)) & (df["open"] > df["close"].shift(1))
    return all_bear & lower_closes & opens_within_prev_body


# Registry: name -> (detector_fn, implied_direction). None = tested both ways.
CANDLESTICK_PATTERNS = {
    "bullish_engulfing": (bullish_engulfing, "long"),
    "bearish_engulfing": (bearish_engulfing, "short"),
    "hammer": (hammer, "long"),
    "shooting_star": (shooting_star, "short"),
    "morning_star": (morning_star, "long"),
    "evening_star": (evening_star, "short"),
    "three_white_soldiers": (three_white_soldiers, "long"),
    "three_black_crows": (three_black_crows, "short"),
    "doji": (doji, None),
}
