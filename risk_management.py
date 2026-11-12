"""
risk_management.py
ATR calculation and stop-loss / take-profit level generation.

Two SL/TP approaches are provided:
  1. Structure-based: stop beyond the relevant order block / recent swing
     point, target the next opposing swing (liquidity target).
  2. ATR-based fallback: when no clean structural level is available,
     use a multiple of Average True Range for the stop and a fixed
     reward:risk ratio for the target.

These are mechanical calculations from price structure — not a guarantee
of outcome. Position sizing (how much to risk per trade) is a separate,
equally important decision this module does NOT make for you.
"""

import pandas as pd
import numpy as np


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()


def find_liquidity_targets(df: pd.DataFrame, direction: str, entry_price: float) -> list:
    """
    Returns every opposing swing point beyond entry_price, ordered nearest
    to farthest -- these are the real, structure-backed candidates for a
    take-profit. Farther targets mean bigger reward:risk, but also more
    distance for price to cover (and more chances for the move to fail
    before getting there).
    """
    if direction == "long":
        candidates = df[df["swing_high"] & (df["high"] > entry_price)]
        return sorted(candidates["high"].tolist())  # nearest (smallest) first
    else:
        candidates = df[df["swing_low"] & (df["low"] < entry_price)]
        return sorted(candidates["low"].tolist(), reverse=True)  # nearest (largest) first


def breakeven_win_rate(reward_risk_ratio: float) -> float:
    """The win rate needed just to break even at a given reward:risk ratio.
    E.g. at 1:5, you only need to win 1 in 6 trades (16.7%) to break even --
    useful context when deciding whether a stretch target is worth taking."""
    if reward_risk_ratio is None or reward_risk_ratio <= 0:
        return None
    return round(1 / (1 + reward_risk_ratio), 3)


def structural_stop_target(df: pd.DataFrame, direction: str, entry_price: float,
                            reward_risk: float = 2.0, max_reward_risk: float = 20.0,
                            min_reward_risk: float = 1.5) -> dict:
    """
    direction: 'long' or 'short'

    Stop placement: unchanged -- beyond the relevant order block / recent
    swing point (see below).

    Target placement: actively searches every real opposing swing point
    (liquidity target) beyond entry, and takes the FARTHEST one whose
    reward:risk still falls within [min_reward_risk, max_reward_risk] --
    i.e. it reaches for the best defensible reward:risk it can find
    (anywhere from roughly 1:1.5 up to 1:20), rather than always snapping to
    the nearest liquidity. If no real liquidity target reaches even
    min_reward_risk, it falls back to a fixed reward_risk multiple of the
    stop distance (labeled 'atr_multiple' in the result, since it isn't
    backed by an actual structure point).

    Important honesty note: reaching further for a bigger reward:risk
    almost always means a LOWER chance of getting there -- price has more
    room to reverse before hitting a farther target. This function does
    not know your win probability; see breakeven_win_rate() to sanity-check
    whether a given ratio is worth it against your own backtest win rate.
    """
    atr = compute_atr(df).iloc[-1]
    if pd.isna(atr):
        atr = df["close"].iloc[-20:].std()  # crude fallback if not enough bars for ATR

    if direction == "long":
        # Only a bullish order block whose zone still sits BELOW current price
        # is a valid stop reference -- one price has already moved past (now
        # above it) is stale and would produce a stop above entry.
        candidate_obs = df[(df["order_block"] == "bullish_ob") & (df["high"] < entry_price)]
        recent_ob = candidate_obs.tail(1)
        if not recent_ob.empty:
            stop = recent_ob["low"].iloc[0] - atr * 0.25
        else:
            candidate_lows = df[df["swing_low"] & (df["low"] < entry_price)]
            recent_low = candidate_lows.tail(1)
            stop = (recent_low["low"].iloc[0] if not recent_low.empty else entry_price - atr * 1.5) - atr * 0.1

    elif direction == "short":
        # Mirror of the long case: only a bearish order block still ABOVE
        # current price is a valid (untested) stop reference.
        candidate_obs = df[(df["order_block"] == "bearish_ob") & (df["low"] > entry_price)]
        recent_ob = candidate_obs.tail(1)
        if not recent_ob.empty:
            stop = recent_ob["high"].iloc[0] + atr * 0.25
        else:
            candidate_highs = df[df["swing_high"] & (df["high"] > entry_price)]
            recent_high = candidate_highs.tail(1)
            stop = (recent_high["high"].iloc[0] if not recent_high.empty else entry_price + atr * 1.5) + atr * 0.1
    else:
        raise ValueError("direction must be 'long' or 'short'")

    risk = abs(entry_price - stop)
    if risk <= 0:
        risk = atr * 0.5  # guard against a degenerate zero-risk stop

    liquidity_targets = find_liquidity_targets(df, direction, entry_price)
    max_distance = risk * max_reward_risk
    min_distance = risk * min_reward_risk

    # Every real liquidity point whose reward:risk falls in the window we'll accept.
    reachable = [
        t for t in liquidity_targets
        if min_distance <= abs(t - entry_price) <= max_distance
    ]

    conservative_target = liquidity_targets[0] if liquidity_targets else None

    if reachable:
        # Farthest qualifying real liquidity point -> best defensible reward:risk.
        target = reachable[-1]
        target_source = "liquidity"
    elif liquidity_targets and abs(liquidity_targets[0] - entry_price) > max_distance:
        # Nearest real liquidity is already beyond our cap -- use it anyway,
        # it's still the most honest read, just capped for sanity by the caller if desired.
        target = liquidity_targets[0]
        target_source = "liquidity_beyond_cap"
    else:
        # No usable real liquidity target at all -- fall back to a fixed
        # multiple of the stop distance. Clearly not liquidity-backed.
        target = entry_price + risk * reward_risk if direction == "long" else entry_price - risk * reward_risk
        target_source = "atr_multiple"

    reward = abs(target - entry_price)
    actual_rr = reward / risk if risk > 0 else np.nan

    conservative_rr = None
    if conservative_target is not None:
        conservative_rr = round(abs(conservative_target - entry_price) / risk, 2) if risk > 0 else None

    # Volatility context: how does current ATR compare to its own recent
    # average? A big spike vs. normal suggests choppier, less reliable
    # structure right now.
    atr_series = compute_atr(df)
    atr_avg = atr_series.tail(50).mean()
    atr_ratio = float(atr / atr_avg) if atr_avg and not pd.isna(atr_avg) else None

    return {
        "entry": round(entry_price, 5),
        "stop_loss": round(stop, 5),
        "take_profit": round(target, 5),
        "target_source": target_source,  # 'liquidity', 'liquidity_beyond_cap', or 'atr_multiple'
        "risk_pips_raw": round(risk, 5),
        "reward_pips_raw": round(reward, 5),
        "reward_risk_ratio": round(actual_rr, 2) if not np.isnan(actual_rr) else None,
        "breakeven_win_rate": breakeven_win_rate(actual_rr if not np.isnan(actual_rr) else None),
        "conservative_target": round(conservative_target, 5) if conservative_target is not None else None,
        "conservative_reward_risk_ratio": conservative_rr,
        "atr": round(float(atr), 5),
        "atr_vs_recent_avg": round(atr_ratio, 2) if atr_ratio is not None else None,
    }


def classify_risk(score: int, reward_risk_ratio: float, atr_vs_recent_avg: float,
                   upcoming_high_impact_events: int, historical_win_rate: float = None,
                   historical_sample_size: int = 0, mtf_agree: int = None,
                   mtf_total: int = None, target_source: str = None) -> dict:
    """
    Combines structure confidence, realized reward:risk, current volatility
    vs. its own recent norm, upcoming news risk, this instrument's own
    historical win rate on similar setups, multi-timeframe confluence, and
    whether the take-profit is backed by real liquidity into a simple
    label. This is a heuristic risk flag, not a guarantee -- always sized
    and confirmed against your own judgment before entering.

    historical_win_rate: fraction (0-1) from backtest.py on this instrument's
    own recent history. None if not enough historical signals to judge.
    mtf_agree / mtf_total: how many of the checked timeframes (1h/30m/15m/
    5m/1m) agree on direction, out of how many had a defined trend.
    target_source: 'liquidity', 'liquidity_beyond_cap', or 'atr_multiple'
    from structural_stop_target -- an 'atr_multiple' target has no real
    structure point backing it, just a fixed multiple of the stop distance.
    """
    risk_points = 0
    flags = []

    if score < 4:
        risk_points += 1
        flags.append("structure confidence below 4/5")
    if reward_risk_ratio is None or reward_risk_ratio < 1.5:
        risk_points += 1
        flags.append("reward:risk below 1.5")
    if atr_vs_recent_avg is not None and atr_vs_recent_avg > 1.5:
        risk_points += 1
        flags.append(f"volatility {atr_vs_recent_avg}x above its recent average (choppy)")
    if upcoming_high_impact_events > 0:
        risk_points += 1
        flags.append(f"{upcoming_high_impact_events} high-impact news event(s) due in this window")
    if target_source == "atr_multiple":
        risk_points += 1
        flags.append("take-profit is not backed by a real liquidity point (no clean target found -- extrapolated)")
    elif target_source == "liquidity_beyond_cap" and reward_risk_ratio and reward_risk_ratio > 20:
        flags.append(f"target reaches a real but distant liquidity point ({reward_risk_ratio}:1) -- long way to travel")

    if historical_sample_size < 5:
        flags.append("not enough historical signals on this instrument yet to judge reliability")
    elif historical_win_rate is not None and historical_win_rate < 0.45:
        risk_points += 1
        flags.append(f"this pattern only won {historical_win_rate:.0%} of the last {historical_sample_size} times on this instrument")
    elif historical_win_rate is not None:
        flags.append(f"this pattern won {historical_win_rate:.0%} of the last {historical_sample_size} times on this instrument")

    if mtf_total:
        agree_ratio = mtf_agree / mtf_total if mtf_total else 0
        if agree_ratio < 0.6:
            risk_points += 1
            flags.append(f"timeframes disagree -- only {mtf_agree}/{mtf_total} aligned")
        else:
            flags.append(f"{mtf_agree}/{mtf_total} timeframes aligned on direction")

    if risk_points == 0:
        label = "Lower risk"
    elif risk_points == 1:
        label = "Moderate risk"
    else:
        label = "Higher risk -- consider skipping, waiting for confirmation, or reducing size"

    return {"label": label, "risk_points": risk_points, "flags": flags}
