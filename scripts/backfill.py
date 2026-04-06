#!/usr/bin/env python3
"""
traumayellow · backfill.py
Backfills signals.csv for dates in ed_counts.csv that are missing from signals.csv.

Run locally (not via GitHub Actions) since it needs to make hundreds of API calls.
Open-Meteo supports full history. AirNow only works for current obs.
Crime data: only last ~12 months available per-year service.
NWSS: only goes back to ~Sep 2025 for Jefferson County.

Usage:
  python scripts/backfill.py                       # all missing dates
  python scripts/backfill.py --start 2024-07-01    # from date
  python scripts/backfill.py --start 2025-01-01 --end 2025-06-30
  python scripts/backfill.py --dry-run             # show what would be done
"""

import argparse
import datetime
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",   help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     help="End date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--dry-run", action="store_true", help="Show dates without collecting")
    parser.add_argument("--delay",   type=float, default=0.5,
                        help="Seconds between API calls (default: 0.5)")
    args = parser.parse_args()

    import csv

    ed_counts_path = ROOT / "data" / "ed_counts.csv"
    signals_path   = ROOT / "data" / "signals.csv"

    if not ed_counts_path.exists():
        print("ed_counts.csv not found — nothing to backfill against")
        sys.exit(1)

    with open(ed_counts_path) as f:
        ed_dates = set(row["date"] for row in csv.DictReader(f))

    existing_dates = set()
    if signals_path.exists():
        with open(signals_path) as f:
            existing_dates = set(row["date"] for row in csv.DictReader(f))

    missing = sorted(ed_dates - existing_dates)

    # Apply date filters
    if args.start:
        missing = [d for d in missing if d >= args.start]
    if args.end:
        missing = [d for d in missing if d <= args.end]
    else:
        # Default: up to yesterday
        yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
        missing = [d for d in missing if d <= yesterday]

    if not missing:
        print("No missing dates to backfill.")
        return

    print(f"Backfilling {len(missing)} dates: {missing[0]} → {missing[-1]}")
    print(f"Estimated time: {len(missing) * args.delay / 60:.1f} min\n")

    if args.dry_run:
        for d in missing[:20]:
            print(f"  would collect: {d}")
        if len(missing) > 20:
            print(f"  ... and {len(missing)-20} more")
        return

    errors = []
    for i, date_str in enumerate(missing):
        print(f"  [{i+1}/{len(missing)}] {date_str}", end="", flush=True)
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "collect_signals.py"),
             "--date", date_str],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f" ❌")
            errors.append(date_str)
            if result.stderr:
                print(f"    {result.stderr.strip()[:120]}")
        else:
            # Extract the summary line
            summary = next((l for l in result.stdout.split('\n') if l.strip().startswith('  weather')), "ok")
            print(f" ✅ {summary.strip()[:80]}")
        time.sleep(args.delay)

    print(f"\nBackfill complete. {len(missing)-len(errors)}/{len(missing)} succeeded.")
    if errors:
        print(f"Failed dates ({len(errors)}): {errors[:10]}")


if __name__ == "__main__":
    main()
