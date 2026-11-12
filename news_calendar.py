"""
news_calendar.py
REAL historical high-impact news events for 2020-2024 -- fixing the gap
flagged directly: every backtest so far has been blind to actual news.

FOMC dates: actual public meeting dates (decision announced on the
  second day of each 2-day meeting). This is real, verifiable public
  record -- cross-check against federalreserve.gov if you want to audit it.

NFP dates: computed by RULE (first Friday of each month, the actual
  publication rule the BLS uses), not hardcoded -- this is accurate by
  construction, with rare BLS holiday-shift exceptions not modeled.

CPI dates: APPROXIMATED as the 12th of each month. Real CPI release
  dates vary (roughly the 10th-15th) and require a source this sandbox
  can't fetch. This is good enough to test "does proximity to a major
  data release affect trade outcomes" directionally, not precise enough
  to trade around exactly -- swap in the real BLS schedule before relying
  on this for anything live.
"""

from datetime import datetime, timedelta
import calendar

import pandas as pd

# Actual FOMC decision dates (2nd day of each meeting), 2020-2024
FOMC_DATES = [
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29", "2020-06-10",
    "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28",
    "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27",
    "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
    "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
    "2024-09-18", "2024-11-07", "2024-12-18",
]


def _first_friday(year: int, month: int) -> datetime:
    first_day = datetime(year, month, 1)
    offset = (4 - first_day.weekday()) % 7  # Friday = weekday 4
    return first_day + timedelta(days=offset)


def get_nfp_dates(start_year: int = 2020, end_year: int = 2024) -> list:
    dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            dates.append(_first_friday(year, month))
    return dates


def get_cpi_dates_approx(start_year: int = 2020, end_year: int = 2024) -> list:
    """Approximate -- see module docstring."""
    dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            dates.append(datetime(year, month, 12))
    return dates


# ---------------------------------------------------------------------------
# OFFICIAL 2026-2027 DATES (added Nov 2026 so the live news filter works today).
# Sources: Federal Reserve FOMC calendar (2027 dates are TENTATIVE), BLS release
# schedules for the Employment Situation (NFP) and CPI. Each event is stored at
# its real release time (Eastern -> converted to UTC), so the 2h buffer is
# centred on the actual release, not on midnight. 2025 is not covered (not
# needed for live use). CHECK THIS FILE AGAIN before the dates run out --
# the live report warns when the calendar is about to end.
# ---------------------------------------------------------------------------
from zoneinfo import ZoneInfo

OFFICIAL_FOMC_2026_27 = [  # decision day, statement at 14:00 ET
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
    "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09", "2027-07-28", "2027-09-15", "2027-10-27", "2027-12-08",
]
OFFICIAL_NFP_2026_27 = [  # 08:30 ET
    "2026-01-09", "2026-02-11", "2026-03-06", "2026-04-03", "2026-05-08", "2026-06-05",
    "2026-07-02", "2026-08-07", "2026-09-04", "2026-10-02", "2026-11-06", "2026-12-04",
    "2027-01-08", "2027-02-05", "2027-03-05", "2027-04-02", "2027-05-07", "2027-06-04",
    "2027-07-02", "2027-08-06", "2027-09-03", "2027-10-08", "2027-11-05", "2027-12-03",
]
OFFICIAL_CPI_2026_27 = [  # 08:30 ET
    "2026-01-13", "2026-02-13", "2026-03-11", "2026-04-10", "2026-05-12", "2026-06-10",
    "2026-07-14", "2026-08-12", "2026-09-11", "2026-10-14", "2026-11-10", "2026-12-10",
    "2027-01-13", "2027-02-11", "2027-03-10", "2027-04-13", "2027-05-12", "2027-06-10",
    "2027-07-14", "2027-08-11", "2027-09-14", "2027-10-14", "2027-11-10", "2027-12-10",
]


def _et_to_naive_utc(date_str: str, hour: int, minute: int) -> datetime:
    local = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=hour, minute=minute, tzinfo=ZoneInfo("America/New_York"))
    return local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)


def get_official_events_2026_27() -> list:
    rows = []
    for d in OFFICIAL_FOMC_2026_27:
        rows.append({"time": _et_to_naive_utc(d, 14, 0), "event": "FOMC decision", "impact": "high", "currency": "USD"})
    for d in OFFICIAL_NFP_2026_27:
        rows.append({"time": _et_to_naive_utc(d, 8, 30), "event": "US Non-Farm Payrolls", "impact": "high", "currency": "USD"})
    for d in OFFICIAL_CPI_2026_27:
        rows.append({"time": _et_to_naive_utc(d, 8, 30), "event": "US CPI", "impact": "high", "currency": "USD"})
    return rows


def last_event_time() -> datetime:
    """Latest event in the calendar (naive UTC) -- used to warn before the calendar runs out."""
    return get_real_events()["time"].max().to_pydatetime()


def get_real_events(start_year: int = 2020, end_year: int = 2027) -> pd.DataFrame:
    """All events combined into one sorted DataFrame with time, event, impact, currency.
    2020-2024: the original calendar (backtests are unchanged). 2026-2027: official dates."""
    rows = []
    legacy_end = min(end_year, 2024)
    for d in FOMC_DATES:
        dt = datetime.strptime(d, "%Y-%m-%d")
        if start_year <= dt.year <= legacy_end:
            rows.append({"time": dt, "event": "FOMC decision", "impact": "high", "currency": "USD"})
    if start_year <= legacy_end:
        for dt in get_nfp_dates(start_year, legacy_end):
            rows.append({"time": dt, "event": "US Non-Farm Payrolls", "impact": "high", "currency": "USD"})
        for dt in get_cpi_dates_approx(start_year, legacy_end):
            rows.append({"time": dt, "event": "US CPI (approx. date)", "impact": "high", "currency": "USD"})
    for row in get_official_events_2026_27():
        if start_year <= row["time"].year <= end_year:
            rows.append(row)
    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    return df


_EVENTS_CACHE = None


def hours_to_nearest_event(timestamp: pd.Timestamp) -> float:
    """Hours between `timestamp` and the nearest known high-impact event, or None if out of range."""
    global _EVENTS_CACHE
    if _EVENTS_CACHE is None:
        _EVENTS_CACHE = get_real_events()

    if _EVENTS_CACHE.empty:
        return None

    ts = pd.Timestamp(timestamp)
    # Real intraday data (yfinance, etc.) often comes back timezone-aware,
    # while these event dates are naive. pandas refuses to subtract a
    # tz-aware and tz-naive timestamp -- normalize both to naive UTC
    # before comparing. Precision to the hour is all this needs.
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)

    diffs = (_EVENTS_CACHE["time"] - ts).abs()
    min_diff = diffs.min()
    return round(min_diff.total_seconds() / 3600, 1)


def events_in_window(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    events = get_real_events()
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start.tzinfo is not None:
        start = start.tz_convert("UTC").tz_localize(None)
    if end.tzinfo is not None:
        end = end.tz_convert("UTC").tz_localize(None)
    return events[(events["time"] >= start) & (events["time"] <= end)]
