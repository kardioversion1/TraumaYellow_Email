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
    # temperature delta (behavioral surge signal)
    "temp_7d_mean", "temp_delta", "temp_surge_flag",
    # air quality (AirNow EPA)
    "aqi_o3", "aqi_pm25", "aqi_overall",
    # ozone lag columns (for WMA in retrain)
    "aqi_o3_lag3", "aqi_o3_lag4", "aqi_o3_lag5",
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
        "temp_max_f":      d["temperature_2m_max"][0],
        "temp_min_f":      d["temperature_2m_min"][0],
        "precip_mm":       d["precipitation_sum"][0] or 0,
        "snowfall_mm":     d["snowfall_sum"][0] or 0,
        "wind_max_mph":    d["wind_speed_10m_max"][0],
        "cloud_cover_pct": d["cloud_cover_mean"][0],
    }


def compute_temp_delta(target: datetime.date, current_temp_max: float) -> dict:
    """
    Fetch the past 7 days of temp_max to compute rolling mean and delta.
    A delta > 15°F flags a behavioral/trauma surge risk.
    """
    start = (target - datetime.timedelta(days=7)).isoformat()
    end   = (target - datetime.timedelta(days=1)).isoformat()
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        f"&daily=temperature_2m_max"
        f"&temperature_unit=fahrenheit"
        f"&start_date={start}&end_date={end}&timezone=America%2FNew_York"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    past_temps = r.json()["daily"]["temperature_2m_max"]
    past_temps = [t for t in past_temps if t is not None]
    if not past_temps:
        return {"temp_7d_mean": None, "temp_delta": None, "temp_surge_flag": 0}
    mean_7d = sum(past_temps) / len(past_temps)
    delta   = current_temp_max - mean_7d
    return {
        "temp_7d_mean":    round(mean_7d, 1),
        "temp_delta":      round(delta, 1),
        "temp_surge_flag": int(delta > 15),
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


def fetch_ozone_lags(target: datetime.date) -> dict:
    """
    Fetch aqi_o3 for 3, 4, 5 days ago so retrain.py can compute the WMA
    without needing to look backwards in signals.csv.
    These are pre-computed here for convenience.
    """
    result = {"aqi_o3_lag3": None, "aqi_o3_lag4": None, "aqi_o3_lag5": None}
    if not AIRNOW_KEY:
        return result
    for lag, key in [(3, "aqi_o3_lag3"), (4, "aqi_o3_lag4"), (5, "aqi_o3_lag5")]:
        lag_date = target - datetime.timedelta(days=lag)
        try:
            data = fetch_airnow(lag_date)
            result[key] = data.get("aqi_o3")
        except Exception as e:
            print(f"  [WARN] ozone lag {lag}: {e}", file=sys.stderr)
    return result


# ── CDC NSSP (ER visit % flu/COVID) ───────────────────────────────────────────
def fetch_nssp(date: datetime.date) -> dict:
    week_end = date - datetime.timedelta(days=date.weekday())
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
    ds = (date - datetime.timedelta(days=14)).isoformat()
    url = (
        "https://data.cdc.gov/resource/2ew6-ywp6.json"
        f"?$where=date_end>='{ds}'"
        "&county_fips=21111"
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
    base      = "https://maps.lojic.org/arcgis/rest/services"
    crime_url = f"{base}/LMPD_Crime/MapServer/0"
    ems_url   = f"{base}/EMS/MapServer/0"
    coll_url  = f"{base}/Traffic_Collisions/MapServer/0"
    return {
        "crime_violent_7d": safe_get(lambda: fetch_lojic_count(crime_url, date), "LOJIC crime"),
        "ems_runs_7d":      safe_get(lambda: fetch_lojic_count(ems_url,   date), "LOJIC EMS"),
        "collisions_7d":    safe_get(lambda: fetch_lojic_count(coll_url,  date), "LOJIC collisions"),
    }


# ── Ticketmaster event attendance ─────────────────────────────────────────────
def fetch_events(date: datetime.date) -> dict:
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

    row = {"date": target.isoformat()}
    row["is_holiday"]  = int(target in HOLIDAYS)
    row["day_of_week"] = target.weekday()

    # Weather
    weather = safe_get(lambda: fetch_weather(target), "weather")
    row.update(weather or {})

    # Temperature delta (behavioral surge signal)
    if weather and weather.get("temp_max_f") is not None:
        delta = safe_get(lambda: compute_temp_delta(target, weather["temp_max_f"]), "temp_delta")
        row.update(delta or {})

    # AirNow
    airnow = safe_get(lambda: fetch_airnow(target), "AirNow")
    row.update(airnow or {})

    # Ozone lags (for WMA respiratory signal)
    o3_lags = safe_get(lambda: fetch_ozone_lags(target), "ozone lags")
    row.update(o3_lags or {})

    # CDC surveillance
    nssp = safe_get(lambda: fetch_nssp(target), "NSSP")
    row.update(nssp or {})

    nwss = safe_get(lambda: fetch_nwss(target), "NWSS")
    row.update(nwss or {})

    # LOJIC community signals
    lojic = safe_get(lambda: fetch_lojic(target), "LOJIC")
    row.update(lojic or {})

    # Events
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

    print(f"  temp_delta={row.get('temp_delta')}°F  surge_flag={row.get('temp_surge_flag')}  aqi_o3={row.get('aqi_o3')}")
    print("Done.")


if __name__ == "__main__":
    main()
