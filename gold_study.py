"""
gold_study.py

GOLD STUDY BOT -- runs for 12 months and learns how to trade gold. It does
NOT send trade alerts and it NEVER changes its own rules. Every run it:

  1. Pulls the newest gold candles (5m, 15m, 30m, 1h, 1d; 4h is built from 1h)
     and MERGES them into its own saved history (study/data/*.csv), so after
     a year you have far more intraday history than the free source keeps.
  2. Finds every candlestick pattern, chart pattern, RSI setting and moving-
     average cross on every timeframe, and logs each one as a PAPER TRADE:
     SL = 1.5x ATR, TP = 2:1, 0.02R cost, followed until SL/TP is hit (same
     rules as the earlier gold scan). Only signals AFTER the study start date
     count -- this is a true forward test.
  3. Records the context of each trade: session, weekday, 4h and daily trend
     (with/against the trade), volatility regime, and whether a high-impact
     news event (FOMC/NFP/CPI) was within 2 hours.
  4. Sends you, on Telegram:
        - a MONTHLY report (text you can paste into chat)
        - a QUARTERLY detailed report every 3 months (a file + short summary)
     Reports are due on the same day-of-month the study started.
     A manual run ("Run workflow") sends the current status report.

HONESTY RULES BUILT IN
  - Results are always shown with trade counts. Fewer than 15 trades = "too few".
  - Every pattern/setting is compared to RANDOM entries over the same period.
  - ~170 combinations are tracked at once, so a few will look good by luck.
    A "clear edge" needs 30+ trades AND a result well above random (z >= 2.5).
  - Trades that neither hit SL nor TP within 60 candles are not counted
    (same convention as the earlier scans).

RUN:
    export TELEGRAM_BOT_TOKEN="..."  TELEGRAM_CHAT_ID="..."
    python gold_study.py
"""

import os
import json
import warnings
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests

from structure import find_swing_points
from risk_management import compute_atr
from candlestick_patterns import CANDLESTICK_PATTERNS
from chart_patterns import CHART_PATTERNS
from data_feed import get_candles
from news_calendar import hours_to_nearest_event

warnings.filterwarnings("ignore")

SYMBOL = "XAUUSD"
STUDY_DIR = "study"
DATA_DIR = f"{STUDY_DIR}/data"
REPORT_DIR = f"{STUDY_DIR}/reports"
TRADES_PATH = f"{STUDY_DIR}/trades.csv"
STATE_PATH = f"{STUDY_DIR}/state.json"
STUDY_MONTHS = 12

SL_ATR_MULT = 1.5
REWARD_RISK = 2.0
MAX_HOLDING = 60
COST_R = 0.02
NEWS_BUFFER_HOURS = 2.0
MIN_N = 15            # below this a result is "too few trades"
CLEAR_EDGE_N = 30
CLEAR_EDGE_Z = 2.5

TF_MINUTES = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
STUDY_TFS = ["5m", "15m", "30m", "1h", "4h", "1d"]
BASE_TFS = ["5m", "15m", "30m", "1h", "1d"]
# candle counts requested from the free source (converted to days inside get_candles)
FETCH = {"5m": 17280, "15m": 5760, "30m": 2880, "1h": 24 * 700, "1d": 24 * 900}
# each run re-checks only the newest bars (context bars are used for detection only)
WINDOW = {"5m": 3000, "15m": 1500, "30m": 1000, "1h": 1000, "4h": 600, "1d": 100000}
CONTEXT = 200

RSI_PERIODS = [7, 14, 21]
RSI_LEVELS = [(20, 80), (25, 75), (30, 70)]
MA_FAST = [5, 10, 20]
MA_SLOW = [30, 50, 100]

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
def send_text(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    ok = True
    for i in range(0, len(text), 3900):
        try:
            r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                              json={"chat_id": TELEGRAM_CHAT_ID, "text": text[i:i + 3900]}, timeout=30)
            ok = ok and r.ok
        except Exception:
            ok = False
    return ok


def send_document(path: str, caption: str = "") -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        with open(path, "rb") as f:
            r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument",
                              data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption[:1000]},
                              files={"document": f}, timeout=60)
        return r.ok
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Candle history (own saved copy, grows over the year)
# ---------------------------------------------------------------------------
def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["open", "high", "low", "close", "volume"]].copy() if "volume" in df.columns \
        else df[["open", "high", "low", "close"]].assign(volume=0.0)
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    df.index = idx
    df.index.name = "time"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["open", "high", "low", "close"])


def _complete_only(df: pd.DataFrame, tf: str, now: pd.Timestamp) -> pd.DataFrame:
    """Drop the candle that is still forming -- it would be stored half-finished."""
    return df[df.index + pd.Timedelta(minutes=TF_MINUTES[tf]) <= now]


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time").sort_index()


def _build_4h(h1: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    if h1.empty:
        return h1
    h4 = h1.resample("4h").agg({"open": "first", "high": "max", "low": "min",
                                 "close": "last", "volume": "sum"}).dropna(subset=["open"])
    return _complete_only(h4, "4h", now)


def update_history(now: pd.Timestamp, fetch_fn=None) -> tuple:
    """Fetch newest candles, merge into saved history. Returns (frames, errors)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    fetch_fn = fetch_fn or (lambda tf: get_candles(SYMBOL, interval=tf, lookback=FETCH[tf], source="yfinance"))
    frames, errors = {}, {}
    for tf in BASE_TFS:
        path = f"{DATA_DIR}/{SYMBOL}_{tf}.csv"
        stored = _load_csv(path)
        try:
            new = _complete_only(_normalize(fetch_fn(tf)), tf, now)
            merged = pd.concat([stored, new]) if not stored.empty else new
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            merged.to_csv(path)
            frames[tf] = merged
        except Exception as exc:
            errors[tf] = str(exc)
            if not stored.empty:
                frames[tf] = stored
    if not frames:
        raise RuntimeError(f"No gold data could be loaded: {errors}")
    if "1h" in frames:
        frames["4h"] = _build_4h(frames["1h"], now)
    return frames, errors


# ---------------------------------------------------------------------------
# Paper-trade simulation
# ---------------------------------------------------------------------------
def simulate_np(high, low, close, atr, positions, dirs) -> list:
    """Returns [(position, r_multiple)] for trades that hit SL or TP within MAX_HOLDING candles."""
    n = len(close)
    out = []
    for i, d in zip(positions, dirs):
        if i + 1 >= n:
            continue
        a = atr[i]
        if not (a > 0):
            continue
        entry = close[i]
        risk = a * SL_ATR_MULT
        if d == 1:
            sl, tp = entry - risk, entry + risk * REWARD_RISK
        else:
            sl, tp = entry + risk, entry - risk * REWARD_RISK
        for j in range(i + 1, min(i + MAX_HOLDING, n - 1) + 1):
            if d == 1:
                if low[j] <= sl:
                    out.append((i, -1.0 - COST_R)); break
                if high[j] >= tp:
                    out.append((i, REWARD_RISK - COST_R)); break
            else:
                if high[j] >= sl:
                    out.append((i, -1.0 - COST_R)); break
                if low[j] <= tp:
                    out.append((i, REWARD_RISK - COST_R)); break
    return out


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / period, adjust=False).mean()
    al = loss.ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + ag / al.replace(0, np.nan))


def _trend_series(df: pd.DataFrame, tf: str) -> pd.Series:
    """'up' / 'down' / 'mixed' per bar (close vs EMA50 and its slope), indexed by the
    time each bar CLOSED -- so looking it up at a trade's entry time never peeks ahead."""
    ema = df["close"].ewm(span=50, adjust=False).mean()
    s = pd.Series("mixed", index=df.index, dtype=object)
    s[(df["close"] > ema) & (ema > ema.shift(5))] = "up"
    s[(df["close"] < ema) & (ema < ema.shift(5))] = "down"
    s.iloc[:60] = "unknown"
    s.index = s.index + pd.Timedelta(minutes=TF_MINUTES[tf])
    return s


def _session(hour: int) -> str:
    if hour < 7:
        return "Asia (00-07 UTC)"
    if hour < 12:
        return "London (07-12 UTC)"
    if hour < 16:
        return "London/NY overlap (12-16 UTC)"
    if hour < 21:
        return "New York (16-21 UTC)"
    return "Late (21-24 UTC)"


def _align(trend, direction):
    if trend in (None, "unknown") or pd.isna(trend):
        return "unknown"
    if trend == "mixed":
        return "mixed"
    return "with" if (trend == "up") == (direction == "long") else "against"


def _signal_sets(tail: pd.DataFrame):
    """Yield (kind, name, positions array, directions array) for every pattern / RSI / MA signal."""
    n = len(tail)
    for pname, (fn, d) in {**CANDLESTICK_PATTERNS, **CHART_PATTERNS}.items():
        try:
            sig = fn(tail)
        except Exception:
            continue
        pos = np.where(pd.Series(sig).fillna(False).astype(bool).values)[0]
        for direction in (["long", "short"] if d is None else [d]):
            yield "pattern", pname, pos, np.full(len(pos), 1 if direction == "long" else -1)
    close = tail["close"]
    for period in RSI_PERIODS:
        r = _rsi(close, period)
        for lo, hi in RSI_LEVELS:
            up = ((r <= lo) & (r.shift(1) > lo)).values
            dn = ((r >= hi) & (r.shift(1) < hi)).values
            pos = np.where(up | dn)[0]
            yield "rsi", f"RSI{period} ({lo}/{hi})", pos, np.where(up[pos], 1, -1)
    for fast in MA_FAST:
        for slow in MA_SLOW:
            f, s = close.rolling(fast).mean(), close.rolling(slow).mean()
            up = ((f > s) & (f.shift(1) <= s.shift(1))).values
            dn = ((f < s) & (f.shift(1) >= s.shift(1))).values
            pos = np.where(up | dn)[0]
            yield "ma", f"MA {fast}/{slow} cross", pos, np.where(up[pos], 1, -1)


def find_trades(tf: str, df: pd.DataFrame, trends: dict, start: pd.Timestamp) -> list:
    """All paper trades for one timeframe whose signal is after `start`, newest window only."""
    if len(df) < CONTEXT + 50:
        return []
    tail = find_swing_points(df.iloc[-(WINDOW[tf] + CONTEXT):].copy(), left=3, right=3)
    n = len(tail)
    dur = pd.Timedelta(minutes=TF_MINUTES[tf])
    entry_times = tail.index + dur
    allowed = (np.arange(n) >= max(CONTEXT, n - WINDOW[tf])) & (entry_times >= start)
    if not allowed.any():
        return []
    high, low, close = tail["high"].values, tail["low"].values, tail["close"].values
    atr_s = compute_atr(tail)
    atr = atr_s.values
    atr_ratio = (atr_s / atr_s.rolling(100).mean()).values

    rows = []
    for kind, name, pos, dirs in _signal_sets(tail):
        keep = allowed[pos]
        pos, dirs = pos[keep], dirs[keep]
        if len(pos) == 0:
            continue
        res = simulate_np(high, low, close, atr, pos, dirs)
        dmap = dict(zip(pos.tolist(), dirs.tolist()))
        for i, r in res:
            direction = "long" if dmap[i] == 1 else "short"
            et = entry_times[i]
            ratio = atr_ratio[i]
            regime = "unknown" if np.isnan(ratio) else ("calm" if ratio < 0.8 else "high" if ratio > 1.3 else "normal")
            t4 = trends["4h"].asof(et) if "4h" in trends else None
            t1d = trends["1d"].asof(et) if "1d" in trends else None
            hrs = hours_to_nearest_event(et)
            rows.append({
                "kind": kind, "name": name, "tf": tf, "direction": direction,
                "signal_time": tail.index[i].isoformat(), "entry_time": et.isoformat(), "r": round(r, 3),
                "session": _session(et.hour), "dow": et.day_name(),
                "align4h": _align(t4, direction), "align1d": _align(t1d, direction),
                "atr_regime": regime,
                "near_news": bool(hrs is not None and hrs <= NEWS_BUFFER_HOURS),
            })
    return rows


def update_trades(frames: dict, start: pd.Timestamp) -> pd.DataFrame:
    """Re-check the newest bars of every timeframe and merge new/resolved trades into the log."""
    trends = {tf: _trend_series(frames[tf], tf) for tf in ("4h", "1d") if tf in frames and len(frames[tf]) > 60}
    new_rows = []
    for tf in STUDY_TFS:
        if tf in frames:
            new_rows.extend(find_trades(tf, frames[tf], trends, start))
    old = pd.read_csv(TRADES_PATH) if os.path.exists(TRADES_PATH) else pd.DataFrame()
    new = pd.DataFrame(new_rows)
    allr = pd.concat([old, new], ignore_index=True) if not old.empty else new
    if allr.empty:
        return allr
    allr = allr.drop_duplicates(subset=["kind", "name", "tf", "direction", "signal_time"], keep="last")
    allr = allr.sort_values("signal_time").reset_index(drop=True)
    os.makedirs(STUDY_DIR, exist_ok=True)
    allr.to_csv(TRADES_PATH, index=False)
    return allr


# ---------------------------------------------------------------------------
# Random-entry baseline (same SL/TP, same period)
# ---------------------------------------------------------------------------
def random_baselines(frames: dict, start: pd.Timestamp, n: int = 300) -> dict:
    """{(tf, 'long'/'short'): mean R of random entries since `start`}."""
    rng = np.random.default_rng(7)
    out = {}
    for tf in STUDY_TFS:
        df = frames.get(tf)
        if df is None or len(df) < 120:
            continue
        times = df.index + pd.Timedelta(minutes=TF_MINUTES[tf])
        cand = np.where((times >= start) & (np.arange(len(df)) >= 30))[0]
        if len(cand) < 60:
            continue
        atr = compute_atr(df).values
        h, l, c = df["high"].values, df["low"].values, df["close"].values
        for direction, d in (("long", 1), ("short", -1)):
            pick = np.sort(rng.choice(cand, size=min(n, len(cand)), replace=False))
            res = simulate_np(h, l, c, atr, pick, np.full(len(pick), d))
            if len(res) >= 30:
                out[(tf, direction)] = float(np.mean([r for _, r in res]))
    return out


# ---------------------------------------------------------------------------
# Statistics + reports
# ---------------------------------------------------------------------------
def _stats(r: pd.Series) -> dict:
    n = len(r)
    return {"n": n, "win": float((r > 0).mean()) if n else 0.0, "exp": float(r.mean()) if n else 0.0,
            "total": float(r.sum()), "std": float(r.std(ddof=1)) if n > 1 else 0.0}


def combo_table(trades: pd.DataFrame, base: dict) -> pd.DataFrame:
    rows = []
    for (kind, name, tf, direction), g in trades.groupby(["kind", "name", "tf", "direction"]):
        s = _stats(g["r"])
        b = base.get((tf, direction))
        edge = s["exp"] - b if b is not None else None
        z = edge / (s["std"] / np.sqrt(s["n"])) if (b is not None and s["n"] > 1 and s["std"] > 0) else None
        if s["n"] < MIN_N:
            flag = "too few"
        elif s["n"] >= CLEAR_EDGE_N and z is not None and z >= CLEAR_EDGE_Z and s["exp"] > 0:
            flag = "CLEAR EDGE"
        elif z is not None and z >= 1.5 and s["exp"] > 0:
            flag = "leaning +"
        elif s["exp"] < 0:
            flag = "losing"
        else:
            flag = "no edge"
        rows.append({"kind": kind, "name": name, "tf": tf, "dir": direction, **s,
                     "base": b, "edge": edge, "z": z, "flag": flag})
    return pd.DataFrame(rows)


def _row(label: str, s: dict) -> str:
    return f"  {label:<34} n={s['n']:<4} win {s['win']*100:4.0f}%  {s['exp']:+.3f}R"


def _cut(trades: pd.DataFrame, col: str, title: str, order=None) -> list:
    lines = [title]
    if trades.empty:
        return lines + ["  (no trades yet)"]
    groups = {k: g for k, g in trades.groupby(col)}
    for k in (order or sorted(groups)):
        if k in groups:
            lines.append(_row(str(k), _stats(groups[k]["r"])))
    return lines


def _fmt_combo(r) -> str:
    base = f"  base {r['base']:+.2f}" if r["base"] is not None else ""
    return (f"  {r['name']} {r['tf']} {r['dir']}: n={int(r['n'])} win {r['win']*100:.0f}% "
            f"{r['exp']:+.3f}R{base} [{r['flag']}]")


def build_report(trades: pd.DataFrame, base: dict, frames: dict, start: pd.Timestamp, now: pd.Timestamp,
                 title: str, window_trades: pd.DataFrame = None, detailed: bool = False) -> str:
    L = [title, "=" * 52]
    days = max(0, (now - start).days)
    L.append(f"Study day {days} of ~{STUDY_MONTHS * 30}. Gold ({SYMBOL}), forward paper trades only.")
    L.append("Candles saved: " + ", ".join(f"{tf} {len(frames[tf])}" for tf in STUDY_TFS if tf in frames))
    pat = trades[trades["kind"] == "pattern"] if not trades.empty else trades
    if trades.empty:
        L.append("\nNo resolved paper trades yet -- check back after a few more days.")
        return "\n".join(L)
    L.append(f"Trades so far: {len(trades)} ({len(pat)} pattern, {len(trades) - len(pat)} RSI/MA).")
    L.append("Rules: SL 1.5x ATR, TP 2:1, 0.02R cost. R = profit measured in units of risk.")
    if window_trades is not None:
        L.append("\nTHIS PERIOD ONLY")
        L.append(_row("all pattern trades", _stats(window_trades[window_trades["kind"] == "pattern"]["r"])))
        L.append(_row("all RSI/MA trades", _stats(window_trades[window_trades["kind"] != "pattern"]["r"])))

    table = combo_table(trades, base)
    enough = table[table["n"] >= MIN_N]
    L.append("\nOVERALL (all patterns, since study start)")
    L.append(_row("all pattern trades", _stats(pat["r"])))
    if base:
        L.append(f"  random-entry average on same data: {np.mean(list(base.values())):+.3f}R")

    pats = enough[enough["kind"] == "pattern"].sort_values("exp", ascending=False)
    top_n, bot_n = (25, 10) if detailed else (8, 3)
    L.append(f"\nBEST PATTERNS (n>={MIN_N}) -- top {top_n}")
    L += [_fmt_combo(r) for _, r in pats.head(top_n).iterrows()] or ["  (none with enough trades yet)"]
    L.append(f"WORST PATTERNS -- bottom {bot_n}")
    L += [_fmt_combo(r) for _, r in pats.tail(bot_n).iloc[::-1].iterrows()] if len(pats) > top_n else ["  (see above)"]
    few = int((table[table["kind"] == "pattern"]["n"] < MIN_N).sum()) if not table.empty else 0
    L.append(f"  ({few} pattern setups still have fewer than {MIN_N} trades)")

    L.append("\nBEST TIMEFRAME (all patterns)")
    L += _cut(pat, "tf", "", order=STUDY_TFS)[1:]

    ind = enough[enough["kind"] != "pattern"].sort_values("exp", ascending=False)
    L.append(f"\nRSI / MOVING-AVERAGE SETTINGS (n>={MIN_N}) -- best {12 if detailed else 5}")
    L += [_fmt_combo(r) for _, r in ind.head(12 if detailed else 5).iterrows()] or ["  (none with enough trades yet)"]
    if detailed:
        L.append(f"worst 5")
        L += [_fmt_combo(r) for _, r in ind.tail(5).iloc[::-1].iterrows()]

    L += [""] + _cut(pat, "session", "BEST SESSION (all patterns)")
    L += [""] + _cut(pat, "align4h", "WITH / AGAINST THE 4H TREND", order=["with", "mixed", "against", "unknown"])
    L += [""] + _cut(pat, "align1d", "WITH / AGAINST THE DAILY TREND", order=["with", "mixed", "against", "unknown"])
    L += [""] + _cut(pat, "atr_regime", "VOLATILITY AT ENTRY", order=["calm", "normal", "high", "unknown"])
    pn = pat.copy()
    pn["news"] = np.where(pn["near_news"], "within 2h of news", "no news nearby")
    L += [""] + _cut(pn, "news", "NEWS EFFECT")
    L += [""] + _cut(pat, "dow", "WEEKDAY", order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"])

    if detailed and not pats.empty:
        L.append("\nCONSISTENCY -- expectancy by study month for the top 12 patterns (a real edge should repeat)")
        t = pat.copy()
        t["month"] = ((pd.to_datetime(t["signal_time"], utc=True) - start).dt.days // 30 + 1).astype(int)
        for _, r in pats.head(12).iterrows():
            g = t[(t["name"] == r["name"]) & (t["tf"] == r["tf"]) & (t["direction"] == r["dir"])]
            parts = [f"M{m}: {g[g['month'] == m]['r'].mean():+.2f} (n={int((g['month'] == m).sum())})"
                     for m in sorted(g["month"].unique())]
            L.append(f"  {r['name']} {r['tf']} {r['dir']}: " + "  ".join(parts))
    if detailed:
        L.append("\nALL SETUPS WITH n>=10 (pattern, RSI, MA), sorted by expectancy")
        every = table[table["n"] >= 10].sort_values("exp", ascending=False)
        L += [_fmt_combo(r) for _, r in every.iterrows()]

    L.append("\nHOW TO READ THIS")
    L.append(f"  CLEAR EDGE = n>={CLEAR_EDGE_N} and well above random entries (z>={CLEAR_EDGE_Z}).")
    L.append("  ~170 setups are tracked at once, so a few look good by luck -- trust ones that")
    L.append("  stay positive month after month. Session times are fixed UTC hours.")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# State + schedule
# ---------------------------------------------------------------------------
def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_state(state: dict):
    os.makedirs(STUDY_DIR, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def months_elapsed(start: datetime, now: datetime) -> int:
    m = (now.year - start.year) * 12 + (now.month - start.month)
    return m - (1 if now.day < start.day else 0)


def month_start(start: datetime, k: int) -> pd.Timestamp:
    """Date k months after start (day clipped to the month length)."""
    y, m = divmod(start.month - 1 + k, 12)
    year, month = start.year + y, m + 1
    day = min(start.day, pd.Timestamp(year=year, month=month, day=1).days_in_month)
    return pd.Timestamp(year=year, month=month, day=day, hour=start.hour, minute=start.minute, tz="UTC")


def send_report(text: str, name: str, caption: str = ""):
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = f"{REPORT_DIR}/{name}.txt"
    with open(path, "w") as f:
        f.write(text)
    return path


def run(now: datetime = None, on_request: bool = None, fetch_fn=None):
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    if on_request is None:
        on_request = os.environ.get("EVENT_NAME", "") == "workflow_dispatch"
    state = load_state()
    first_run = "start" not in state
    if first_run:
        state.update({"start": now.isoformat(), "reports_sent": 0})
    start = pd.Timestamp(state["start"])

    frames, errors = update_history(now, fetch_fn)
    trades = update_trades(frames, start)
    if trades.empty:
        trades = pd.DataFrame(columns=["kind", "name", "tf", "direction", "signal_time", "r"])
    print(f"History: " + ", ".join(f"{tf}={len(df)}" for tf, df in frames.items()))
    print(f"Paper trades logged: {len(trades)}   errors: {errors or 'none'}")

    if first_run:
        send_text("🥇 Gold study bot started.\nIt will log every pattern, RSI and moving-average signal on gold as a "
                  "paper trade for 12 months, and send a monthly report (and a detailed one every 3 months).\n"
                  "Candles loaded: " + ", ".join(f"{tf} {len(df)}" for tf, df in frames.items()) +
                  (f"\n⚠️ Data problems: {errors}" if errors else ""))

    base = random_baselines(frames, start) if not trades.empty else {}

    if on_request and not first_run:
        send_text(build_report(trades, base, frames, start, now, "📲 GOLD STUDY -- status on request"))

    sent = int(state.get("reports_sent", 0))
    due = min(months_elapsed(start.to_pydatetime(), now.to_pydatetime()), STUDY_MONTHS)
    while sent < due:
        sent += 1
        end = month_start(start.to_pydatetime(), sent)
        window = pd.to_datetime(trades["signal_time"], utc=True) if not trades.empty else pd.Series(dtype="datetime64[ns, UTC]")
        in_month = trades[(window >= month_start(start.to_pydatetime(), sent - 1)) & (window < end)] if not trades.empty else trades
        text = build_report(trades, base, frames, start, now, f"🥇 GOLD STUDY -- MONTH {sent} OF {STUDY_MONTHS}",
                            window_trades=in_month)
        send_report(text, f"month-{sent:02d}")
        send_text(text)
        if sent % 3 == 0:
            q_start = month_start(start.to_pydatetime(), sent - 3)
            qt = trades[(window >= q_start) & (window < end)] if not trades.empty else trades
            qtext = build_report(qt, random_baselines(frames, q_start) if not qt.empty else {}, frames, q_start, now,
                                 f"🥇 GOLD STUDY -- QUARTERLY DETAILED REPORT (months {sent - 2}-{sent})", detailed=True)
            path = send_report(qtext, f"quarter-{sent // 3}")
            send_text(f"📘 Quarterly detailed report (months {sent - 2}-{sent}) attached. Send the file to Claude "
                      f"to discuss how to trade gold.")
            send_document(path, f"Gold study -- quarter {sent // 3}")
        if sent == STUDY_MONTHS:
            send_text("✅ The 12-month gold study is complete. The bot keeps logging but sends no more reports.")
    state["reports_sent"] = sent
    state["last_run"] = now.isoformat()
    save_state(state)
    print(f"Done. Reports sent so far: {sent}")


if __name__ == "__main__":
    run()
