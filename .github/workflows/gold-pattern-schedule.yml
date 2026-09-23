"""
gold_pattern_scheduled.py

Same "only alert on change" behavior as gold_pattern_monitor.py, but
built for GitHub Actions instead of an always-on server. GitHub Actions
runs are stateless -- each run starts fresh with no memory of the last
one -- so this script saves what it saw to a small JSON file
(state/gold_pattern_state.json) and the workflow commits that file back
to the repo after each run. The NEXT run reads it back, so the
"is this new?" comparison survives between runs even though each run is
otherwise a brand new process.

RUN (locally or in CI):
    export TELEGRAM_BOT_TOKEN="..."
    export TELEGRAM_CHAT_ID="..."
    python gold_pattern_scheduled.py
"""

import os
import json
from datetime import datetime, timezone

import requests

from gold_pattern_live import check_gold_live, format_gold_report

STATE_PATH = "state/gold_pattern_state.json"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def send_telegram(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    ok = True
    for i in range(0, len(text), 3900):
        chunk = text[i:i + 3900]
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": f"```\n{chunk}\n```", "parse_mode": "Markdown"},
                timeout=20,
            )
            if not resp.ok:
                print(f"  Telegram send failed: {resp.status_code} {resp.text[:200]}")
                ok = False
        except Exception as exc:
            print(f"  Telegram send failed: {exc}")
            ok = False
    return ok


def signal_fingerprint(result: dict) -> list:
    """List (not set -- has to be JSON-serializable) of the currently active signals."""
    fp = []
    for d in result["detections"]:
        if "error" in d:
            continue
        fp.append([d["pattern"], d["timeframe"], d["direction"], str(d["fire_time"])])
    return fp


def load_previous_state() -> list:
    if not os.path.exists(STATE_PATH):
        return []
    try:
        with open(STATE_PATH) as f:
            return json.load(f).get("fingerprint", [])
    except Exception as exc:
        print(f"Could not read previous state ({exc}) -- treating as empty")
        return []


def save_state(fingerprint: list):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump({"fingerprint": fingerprint,
                    "saved_at": datetime.now(timezone.utc).isoformat()}, f, indent=2)


if __name__ == "__main__":
    previous = load_previous_state()
    previous_set = {tuple(p) for p in previous}

    result = check_gold_live()
    current = signal_fingerprint(result)
    current_set = {tuple(c) for c in current}

    new_signals = current_set - previous_set

    if new_signals:
        report = format_gold_report(result)
        print(report)
        if send_telegram(report):
            print("\nNew signal(s) -- sent to Telegram")
        else:
            print("\nNew signal(s) found but Telegram send failed or not configured")
    else:
        print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}] No change since last check -- staying quiet")

    save_state(current)
    print(f"State saved ({len(current)} active signal(s))")
