"""
data_feed.py
Pulls OHLC price data for the watched instruments.

Two backends supported:
  1. yfinance  -> free, no API key, good for prototyping (has rate limits / occasional gaps)
  2. Twelve Data -> free tier (800 requests/day), needs a free API key from twelvedata.com

Usage:
    from data_feed import get_candles
    df = get_candles("USDJPY", interval="1h", lookback=300, source="yfinance")
"""

import pandas as pd
import requests

# Map our internal symbol names -> provider-specific tickers
YFINANCE_TICKERS = {
    "XAUUSD": "GC=F",       # Gold futures (close proxy for spot gold)
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "USDCHF": "CHF=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "CAD=X",
    "NZDUSD": "NZDUSD=X",
    "AUDJPY": "AUDJPY=X",
    "GBPJPY": "GBPJPY=X",
    "EURJPY": "EURJPY=X",
    "BTCUSD": "BTC-USD",
    "DXY": "DX-Y.NYB",      # US Dollar Index, useful for gold correlation
}

# Timeframes this project supports, and yfinance's own limits on how far
# back each granularity is available (as of yfinance/Yahoo's public API).
# Twelve Data's free tier has its own, generally longer, lookback limits --
# check twelvedata.com/docs if you switch source.
TIMEFRAMES = ["1h", "30m", "15m", "5m", "1m"]

YFINANCE_INTERVAL_MAP = {
    "1h": "60m",
    "30m": "30m",
    "15m": "15m",
    "5m": "5m",
    "1m": "1m",
}

# Max lookback (days) yfinance allows per interval -- requesting more than
# this returns an error or gets silently truncated.
YFINANCE_MAX_LOOKBACK_DAYS = {
    "1h": 730,
    "30m": 60,
    "15m": 60,
    "5m": 60,
    "1m": 7,
}

TWELVEDATA_SYMBOLS = {
    "XAUUSD": "XAU/USD",
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "USDCHF": "USD/CHF",
    "AUDUSD": "AUD/USD",
    "USDCAD": "USD/CAD",
    "NZDUSD": "NZD/USD",
    "AUDJPY": "AUD/JPY",
    "GBPJPY": "GBP/JPY",
    "EURJPY": "EUR/JPY",
    "BTCUSD": "BTC/USD",
}


def get_candles_yfinance(symbol: str, interval: str = "1h", lookback_days: int = 30) -> pd.DataFrame:
    """Pull candles via yfinance. interval: '1h', '30m', '15m', '5m', '1m', '1d', etc."""
    import yfinance as yf  # imported lazily so the rest of the module works without it installed

    ticker = YFINANCE_TICKERS.get(symbol.upper())
    if not ticker:
        raise ValueError(f"No yfinance ticker mapped for symbol '{symbol}'")

    yf_interval = YFINANCE_INTERVAL_MAP.get(interval, interval)
    max_days = YFINANCE_MAX_LOOKBACK_DAYS.get(interval)
    if max_days is not None and lookback_days > max_days:
        lookback_days = max_days  # silently clamp rather than error -- yfinance would truncate anyway

    df = yf.download(ticker, period=f"{lookback_days}d", interval=yf_interval,
                      progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {symbol} ({ticker}) at interval {interval}")

    # yfinance sometimes returns MultiIndex columns (ticker, field) even for
    # a single symbol -- flatten to plain column names, or downstream code
    # ends up comparing/selecting whole Series where it expects scalars,
    # producing "truth value of a Series is ambiguous" errors far from the
    # actual cause. auto_adjust=False is also pinned explicitly above since
    # yfinance changed its own default, which silently changes close prices.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower)
    df.index.name = "time"
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].dropna()
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df


def get_candles_twelvedata(symbol: str, interval: str = "1h", outputsize: int = 300, api_key: str = "") -> pd.DataFrame:
    """Pull candles via Twelve Data REST API. Needs a free API key from twelvedata.com."""
    td_symbol = TWELVEDATA_SYMBOLS.get(symbol.upper())
    if not td_symbol:
        raise ValueError(f"No Twelve Data symbol mapped for '{symbol}'")

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": td_symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": api_key,
        "format": "JSON",
    }
    resp = requests.get(url, params=params, timeout=15)
    data = resp.json()
    if "values" not in data:
        raise RuntimeError(f"Twelve Data error for {symbol}: {data}")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    df = df.rename(columns={"datetime": "time"})
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    if "volume" in df.columns:
        df["volume"] = df["volume"].astype(float)
    else:
        df["volume"] = 0.0
    return df[["open", "high", "low", "close", "volume"]]


def get_candles(symbol: str, interval: str = "1h", lookback: int = 300, source: str = "yfinance", api_key: str = "") -> pd.DataFrame:
    """Unified entry point. source = 'yfinance' or 'twelvedata'.
    lookback means candle COUNT for twelvedata, but is converted to a rough
    day-count for yfinance (which only accepts a period in days)."""
    if source == "yfinance":
        # Rough candles-per-day for each interval, to convert a desired
        # candle count into a day-count period yfinance understands.
        candles_per_day = {"1h": 24, "30m": 48, "15m": 96, "5m": 288, "1m": 1440}.get(interval, 24)
        lookback_days = max(2, -(-lookback // candles_per_day))  # ceiling division
        return get_candles_yfinance(symbol, interval=interval, lookback_days=lookback_days)
    elif source == "twelvedata":
        return get_candles_twelvedata(symbol, interval=interval, outputsize=lookback, api_key=api_key)
    else:
        raise ValueError(f"Unknown source: {source}")
