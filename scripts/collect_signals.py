#!/usr/bin/env python3
"""
traumayellow · collect_signals.py
Runs daily at 06:00 ET via GitHub Actions.
Fetches all environmental/epidemiological signals for TARGET_DATE (yesterday)
and appends one row to data/signals.csv.

Null-tolerant: if any source fails, that column is written as empty string.
Only hard-fails on file I/O errors.

Required GitHub Secrets:
  AIRNOW_API_KEY  - EPA AirNow API key
"""

import os
import sys
import csv
import json
import datetime
import argparse
import requests
from pathlib import Path

ROOT       = Path(__file__).parent.parent
SIGNALS    = ROOT / "data" / "signals.csv"
LAT, LON   = 38.2527, -85.7585   # Jewish Hospital Downtown, Louisville KY
AIRNOW_KEY = os.environ.get("AIRNOW_API_KEY", "")

COLUMNS = [
    "date",
    # weather (Open-Meteo)
    "temp_max_f", "temp_min_f", "precip_mm", "snowfall_mm",
    "wind_max_mph", "cloud_cover_pct",
    # air quality (AirNow EPA)
    "aqi_o3", "aqi_pm25", "aqi_overall",
    # CDC NSSP (flu/COVID ER visits %)
    "nssp_flu_pct", "nssp_covid_pct",
    # CDC NWSS (wastewater COVID percentile)
    "nwss_percentile",
    # LOJIC ArcGIS (Louisville crime/EMS/collisions)
    "crime_violent_7d", "ems_runs_7d", "collisions_7d",
    # time features
    "is_holiday", "day_of_week",
    # event attendance (Ticketmaster)
    "event_attendance",
]


def safe_get(fn, label):
    """Call fn(), return result or None on any exception."""
    try:
        return fn()
    except Exception as e:
        print(f"  [WARN] {label}: {e}", file=sys.stderr)
        return None


# ── Weather: Open-Meteo (free, no key) ────────────────────────────────────────
def fetch_weather(date: datetime.date) -> dict:
    ds = date.isoformat()
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
        f"snowfall_sum,wind_speed_10m_max,cloud_cover_mean"
        f"&temperature_unit=fahrenheit&wind_speed_unit=mph"
        f"&start_date={ds}&end_date={ds}&timezone=America%2FNew_York"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    d = r.json()["daily"]
    return {
        "temp_max_f":    d["temperature_2m_max"][0],
        "temp_min_f":    d["temperature_2m_min"][0],
        "precip_mm":     d["precipitation_sum"][0] or 0,
        "snowfall_mm":   d["snowfall_sum"][0] or 0,
        "wind_max_mph":  d["wind_speed_10m_max"][0],
        "cloud_cover_pct": d["cloud_cover_mean"][0],
    }


# ── Air quality: AirNow EPA ────────────────────────────────────────────────────
def fetch_airnow(date: datetime.date) -> dict:
    if not AIRNOW_KEY:
        raise ValueError("AIRNOW_API_KEY not set")
    ds = date.strftime("%Y-%m-%dT00")
    url = (
        f"https://www.airnowapi.org/aq/observation/latLong/historical/"
        f"?format=application/json"
        f"&latitude={LAT}&longitude={LON}"
        f"&date={ds}&distance=25&API_KEY={AIRNOW_KEY}"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    result = {"aqi_o3": None, "aqi_pm25": None, "aqi_overall": None}
    overall = 0
    for obs in data:
        param = obs.get("ParameterName", "")
        aqi   = obs.get("AQI", None)
        if "OZONE" in param.upper():
            result["aqi_o3"] = aqi
        elif "PM2.5" in param.upper():
            result["aqi_pm25"] = aqi
        if aqi and aqi > overall:
            overall = aqi
    result["aqi_overall"] = overall or None
    return result


# ── CDC NSSP (ER visit % flu/COVID) ───────────────────────────────────────────
def fetch_nssp(date: datetime.date) -> dict:
    # Kentucky statewide NSSP data via CDC API
    # Returns weekly % of ED visits for flu/COVID
    week_end = date - datetime.timedelta(days=date.weekday())  # most recent Monday
    ds = week_end.strftime("%Y-%m-%dT00:00:00.000")
    url = (
        "https://data.cdc.gov/resource/rdmq-nq56.json"
        f"?$where=week_end>='{ds}'&$limit=20"
        "&geography=Kentucky&$order=week_end DESC"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    rows = r.json()
    result = {"nssp_flu_pct": None, "nssp_covid_pct": None}
    for row in rows:
        cat = row.get("pathogen", "").lower()
        pct = row.get("percent_visits", None)
        if pct is None:
            continue
        pct = float(pct)
        if "influenza" in cat and result["nssp_flu_pct"] is None:
            result["nssp_flu_pct"] = pct
        if "covid" in cat and result["nssp_covid_pct"] is None:
            result["nssp_covid_pct"] = pct
    return result


# ── CDC NWSS (wastewater COVID percentile) ────────────────────────────────────
def fetch_nwss(date: datetime.date) -> dict:
    # Jefferson County wastewater data
    ds = (date - datetime.timedelta(days=14)).isoformat()
    url = (
        "https://data.cdc.gov/resource/2ew6-ywp6.json"
        f"?$where=date_end>='{ds}'"
        "&county_fips=21111"  # Jefferson County, KY
        "&$order=date_end DESC&$limit=5"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    rows = r.json()
    if rows:
        return {"nwss_percentile": rows[0].get("ptc_15d")}
    return {"nwss_percentile": None}


# ── LOJIC ArcGIS: crime, EMS, collisions ─────────────────────────────────────
def fetch_lojic_count(service_url: str, date: datetime.date, days: int = 7) -> int:
    start = (date - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    end   = date.strftime("%Y-%m-%d")
    url = (
        f"{service_url}/query"
        f"?where=DATE_OCC>=DATE+'{start}'+AND+DATE_OCC<=DATE+'{end}'"
        f"&returnCountOnly=true&f=json"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json().get("count", 0)


def fetch_lojic(date: datetime.date) -> dict:
    base = "https://maps.lojic.org/arcgis/rest/services"
    # Violent crime (assaults, robberies, etc.)
    crime_url = f"{base}/LMPD_Crime/MapServer/0"
    # EMS runs
    ems_url   = f"{base}/EMS/MapServer/0"
    # Traffic collisions
    coll_url  = f"{base}/Traffic_Collisions/MapServer/0"

    def _crime():
        return fetch_lojic_count(crime_url, date)
    def _ems():
        return fetch_lojic_count(ems_url, date)
    def _coll():
        return fetch_lojic_count(coll_url, date)

    return {
        "crime_violent_7d": safe_get(_crime, "LOJIC crime"),
        "ems_runs_7d":      safe_get(_ems,   "LOJIC EMS"),
        "collisions_7d":    safe_get(_coll,  "LOJIC collisions"),
    }


# ── Ticketmaster event attendance ─────────────────────────────────────────────
def fetch_events(date: datetime.date) -> dict:
    # Sum estimated attendance for Louisville events on target date
    # Ticketmaster Discovery API (key optional, rate-limited without)
    tm_key = os.environ.get("TICKETMASTER_API_KEY", "")
    if not tm_key:
        return {"event_attendance": None}
    ds  = date.strftime("%Y-%m-%dT00:00:00Z")
    de  = date.strftime("%Y-%m-%dT23:59:59Z")
    url = (
        f"https://app.ticketmaster.com/discovery/v2/events.json"
        f"?apikey={tm_key}&city=Louisville&stateCode=KY"
        f"&startDateTime={ds}&endDateTime={de}&size=50"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    embedded = r.json().get("_embedded", {})
    events = embedded.get("events", [])
    total = 0
    for ev in events:
        venue = ev.get("_embedded", {}).get("venues", [{}])[0]
        cap_str = venue.get("upcomingEvents", {}).get("_total", 0)
        try:
            total += int(cap_str)
        except Exception:
            pass
    return {"event_attendance": total or None}


# ── Holiday check ─────────────────────────────────────────────────────────────
HOLIDAYS = {
    datetime.date(2024, 1, 1), datetime.date(2024, 1, 15),
    datetime.date(2024, 2, 19), datetime.date(2024, 5, 27),
    datetime.date(2024, 7, 4), datetime.date(2024, 9, 2),
    datetime.date(2024, 11, 11), datetime.date(2024, 11, 28),
    datetime.date(2024, 12, 25),
    datetime.date(2025, 1, 1), datetime.date(2025, 1, 20),
    datetime.date(2025, 2, 17), datetime.date(2025, 5, 26),
    datetime.date(2025, 7, 4), datetime.date(2025, 9, 1),
    datetime.date(2025, 11, 11), datetime.date(2025, 11, 27),
    datetime.date(2025, 12, 25),
    datetime.date(2026, 1, 1), datetime.date(2026, 1, 19),
    datetime.date(2026, 2, 16), datetime.date(2026, 5, 25),
    datetime.date(2026, 7, 3), datetime.date(2026, 9, 7),
    datetime.date(2026, 11, 11), datetime.date(2026, 11, 26),
    datetime.date(2026, 12, 25),
}


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Target date YYYY-MM-DD (default: yesterday)")
    args = parser.parse_args()

    if args.date:
        target = datetime.date.fromisoformat(args.date)
    else:
        target = datetime.date.today() - datetime.timedelta(days=1)

    print(f"traumayellow · collect_signals.py · {target}")

    # Check for duplicate
    if SIGNALS.exists():
        existing = set()
        with open(SIGNALS) as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add(row["date"])
        if target.isoformat() in existing:
            print(f"  Already have data for {target} — skipping.")
            return

    # Collect all signals
    row = {"date": target.isoformat()}
    row["is_holiday"]   = int(target in HOLIDAYS)
    row["day_of_week"]  = target.weekday()

    weather = safe_get(lambda: fetch_weather(target), "weather")
    row.update(weather or {})

    airnow = safe_get(lambda: fetch_airnow(target), "AirNow")
    row.update(airnow or {})

    nssp = safe_get(lambda: fetch_nssp(target), "NSSP")
    row.update(nssp or {})

    nwss = safe_get(lambda: fetch_nwss(target), "NWSS")
    row.update(nwss or {})

    lojic = safe_get(lambda: fetch_lojic(target), "LOJIC")
    row.update(lojic or {})

    events = safe_get(lambda: fetch_events(target), "Ticketmaster")
    row.update(events or {})

    # Ensure all columns present
    for col in COLUMNS:
        row.setdefault(col, "")

    # Append to CSV
    SIGNALS.parent.mkdir(exist_ok=True)
    write_header = not SIGNALS.exists()
    with open(SIGNALS, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"  Appended: {row}")
    print("Done.")


if __name__ == "__main__":
    main()
