#!/usr/bin/env python3
"""
traumayellow · backfill.py
One-shot script to backfill signals.csv for any date range where
ed_counts.csv has rows but signals.csv is missing entries.

Open-Meteo supports full historical archive (no key needed).
AirNow supports ~60 days back.
CDC NSSP/NWSS provide full history in one call.
LOJIC supports date range queries.

Usage:
  python scripts/backfill.py                      # backfill all dates in ed_counts not in signals
  python scripts/backfill.py --start 2024-07-01   # backfill from date
  python scripts/backfill.py --start 2024-07-01 --end 2025-01-01

Required env:
  AIRNOW_API_KEY   (optional — skips AQ if missing)
"""

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   help="End date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    # Determine date range to backfill
    import csv
    import pandas as pd

    ed_counts_path = ROOT / "data" / "ed_counts.csv"
    signals_path   = ROOT / "data" / "signals.csv"

    if not ed_counts_path.exists():
        print("ed_counts.csv not found — nothing to backfill against")
        sys.exit(1)

    ed_dates = set(
        pd.read_csv(ed_counts_path, usecols=["date"])["date"].tolist()
    )

    existing_dates = set()
    if signals_path.exists():
        existing_dates = set(
            pd.read_csv(signals_path, usecols=["date"])["date"].tolist()
        )

    missing = sorted(ed_dates - existing_dates)

    if args.start:
        start = datetime.date.fromisoformat(args.start)
        missing = [d for d in missing if d >= args.start]
    if args.end:
        missing = [d for d in missing if d <= args.end]

    if not missing:
        print("No missing dates to backfill.")
        return

    print(f"Backfilling {len(missing)} dates: {missing[0]} → {missing[-1]}")

    for date_str in missing:
        print(f"\n  → {date_str}")
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "collect_signals.py"), "--date", date_str],
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"  [ERROR] collect_signals.py failed for {date_str}")

    print("\nBackfill complete.")


if __name__ == "__main__":
    main()
