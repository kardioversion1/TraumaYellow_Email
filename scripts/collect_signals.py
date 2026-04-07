#!/usr/bin/env python3
"""
traumayellow · collect_signals.py
Runs daily at 06:00 ET via GitHub Actions.
Fetches all signals for TARGET_DATE (yesterday) and appends to data/signals.csv.

Null-tolerant: any source failure writes empty string for that column.
Only hard-fails on file I/O errors.

Sources:
  Open-Meteo   — weather + air quality (free, no key, full history)
  LOJIC ArcGIS — crime (per-year FeatureServer, verified working)
  LOJIC ArcGIS — ROW construction permits (road disruption proxy)
  CDC NSSP     — flu/COVID ER visit % Kentucky
  CDC NWSS     — wastewater COVID percentile Jefferson County
  Ticketmaster — Louisville event attendance
"""

import os, sys, csv, datetime, argparse, requests
from pathlib import Path

ROOT       = Path(__file__).parent.parent
SIGNALS    = ROOT / "data" / "signals.csv"
LAT, LON   = 38.2527, -85.7585   # Jewish Hospital Downtown
ORG        = "79kfd2K6fskCAkyg"  # Louisville Metro ArcGIS org (verified)
LOJIC_BASE = f"https://services1.arcgis.com/{ORG}/arcgis/rest/services"

# Per-year crime service names (verified live, ~1 day lag)
CRIME_SERVICES = {
    2024: "crimedata2024",
    2025: "crime_data_2025",
    2026: "crime_data_2026",
}

COLUMNS = [
    "date",
    # weather
    "temp_max_f", "temp_min_f", "precip_mm", "snowfall_mm",
    "wind_max_mph", "cloud_cover_pct",
    # temp delta (behavioral surge signal)
    "temp_7d_mean", "temp_delta", "temp_surge_flag",
    # air quality (Open-Meteo AQ — free, full history)
    "aqi_pm25", "aqi_o3_ugm3",
    # AirNow EPA (current obs only — historical API broken)
    "aqi_o3", "aqi_pm25_airnow", "aqi_overall",
    # ozone lag columns for WMA respiratory signal
    "aqi_o3_lag3", "aqi_o3_lag4", "aqi_o3_lag5",
    # CDC NSSP + velocity
    "nssp_flu_pct", "nssp_covid_pct", "nssp_rsv_pct",
    "nssp_flu_trend", "nssp_flu_velocity",
    # CDC NWSS
    "nwss_percentile",
    # NWS severe weather alerts
    "nws_alert_count", "power_risk_flag",
    # LOJIC community signals
    "crime_violent_7d", "row_permits_catchment",
    # time + school calendar
    "is_holiday", "day_of_week", "is_school_out", "is_school_night",
    # events
    "event_attendance",
]


def safe_get(fn, label):
    try:
        return fn()
    except Exception as e:
        print(f"  [WARN] {label}: {e}", file=sys.stderr)
        return None


# ── Weather + Air Quality: Open-Meteo ─────────────────────────────────────────
def fetch_weather(date: datetime.date) -> dict:
    """Weather from Open-Meteo forecast/archive API (full historical coverage)."""
    ds = date.isoformat()
    # Use archive endpoint for past dates, forecast for recent
    days_ago = (datetime.date.today() - date).days
    base = "https://archive-api.open-meteo.com/v1/archive" if days_ago > 5 \
           else "https://api.open-meteo.com/v1/forecast"
    url = (
        f"{base}?latitude={LAT}&longitude={LON}"
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


def fetch_air_quality(date: datetime.date) -> dict:
    """PM2.5 and ozone from Open-Meteo AQ API (free, full history, µg/m³)."""
    ds = date.isoformat()
    days_ago = (datetime.date.today() - date).days
    base = "https://air-quality-api.open-meteo.com/v1/air-quality"
    url = (
        f"{base}?latitude={LAT}&longitude={LON}"
        f"&hourly=pm2_5,ozone"
        f"&start_date={ds}&end_date={ds}&timezone=America%2FNew_York"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    hourly = r.json().get("hourly", {})
    # Daily max (peak exposure drives health effects)
    pm25_vals  = [x for x in hourly.get("pm2_5", []) if x is not None]
    ozone_vals = [x for x in hourly.get("ozone", []) if x is not None]
    return {
        "aqi_pm25":    round(max(pm25_vals),  1) if pm25_vals  else None,
        "aqi_o3_ugm3": round(max(ozone_vals), 1) if ozone_vals else None,
    }


def compute_temp_delta(target: datetime.date, temp_max: float) -> dict:
    """7-day rolling mean delta — sudden warmth flags behavioral/trauma surge."""
    start = (target - datetime.timedelta(days=7)).isoformat()
    end   = (target - datetime.timedelta(days=1)).isoformat()
    days_ago = (datetime.date.today() - target).days
    base = "https://archive-api.open-meteo.com/v1/archive" if days_ago > 5 \
           else "https://api.open-meteo.com/v1/forecast"
    url = (
        f"{base}?latitude={LAT}&longitude={LON}"
        f"&daily=temperature_2m_max&temperature_unit=fahrenheit"
        f"&start_date={start}&end_date={end}&timezone=America%2FNew_York"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    past = [t for t in r.json()["daily"]["temperature_2m_max"] if t is not None]
    if not past:
        return {"temp_7d_mean": None, "temp_delta": None, "temp_surge_flag": 0}
    mean_7d = sum(past) / len(past)
    delta   = temp_max - mean_7d
    return {
        "temp_7d_mean":    round(mean_7d, 1),
        "temp_delta":      round(delta,   1),
        "temp_surge_flag": int(delta > 15),
    }


# ── AirNow EPA (current obs only — historical endpoint broken) ─────────────────
def fetch_airnow_current() -> dict:
    """AirNow current observations — only valid for today's collection run."""
    key = os.environ.get("AIRNOW_API_KEY", "")
    if not key:
        return {}
    r = requests.get(
        f"https://www.airnowapi.org/aq/observation/latLong/current/"
        f"?format=application/json&latitude={LAT}&longitude={LON}"
        f"&distance=25&API_KEY={key}",
        timeout=15
    )
    r.raise_for_status()
    data = r.json()
    result = {"aqi_o3": None, "aqi_pm25_airnow": None, "aqi_overall": None}
    overall = 0
    for obs in data:
        param = obs.get("ParameterName", "")
        aqi   = obs.get("AQI")
        if aqi and "OZONE" in param.upper():
            result["aqi_o3"] = aqi
        elif aqi and "PM2.5" in param.upper():
            result["aqi_pm25_airnow"] = aqi
        if aqi and aqi > overall:
            overall = aqi
    result["aqi_overall"] = overall or None
    return result


def fetch_ozone_lags_from_omaq(target: datetime.date) -> dict:
    """Ozone lags 3-5 days back from Open-Meteo AQ for respiratory WMA."""
    result = {"aqi_o3_lag3": None, "aqi_o3_lag4": None, "aqi_o3_lag5": None}
    for lag, key in [(3, "aqi_o3_lag3"), (4, "aqi_o3_lag4"), (5, "aqi_o3_lag5")]:
        lag_date = target - datetime.timedelta(days=lag)
        try:
            aq = fetch_air_quality(lag_date)
            result[key] = aq.get("aqi_o3_ugm3")
        except Exception as e:
            print(f"  [WARN] O3 lag {lag}: {e}", file=sys.stderr)
    return result


# ── LOJIC: Crime (verified working) ──────────────────────────────────────────
# Direct-ED-violence offense types (assault, robbery, homicide, rape)
# These generate actual ED visits — more clinically relevant than NIBRS Group A
# which includes 54k+ records of theft, fraud, etc.
VIOLENT_OFFENSES = (
    "offense_classification='8 ROBBERY' OR "
    "offense_classification='9 AGGRAVATED ASSAULT' OR "
    "offense_classification='11 SIMPLE ASSAULT' OR "
    "offense_classification='4 FORCIBLE RAPE' OR "
    "offense_classification='5 SODOMY FORCE' OR "
    "offense_classification='1 HOMICIDE' OR "
    "offense_classification='10 KIDNAPPING ONLY'"
)

def fetch_lojic_crime(date: datetime.date) -> dict:
    """
    Count direct-ED-violence crimes (assault, robbery, homicide, rape)
    that OCCURRED in the 7 days before date.
    Uses TIMESTAMP date filter on date_occurred (not date_reported).
    Queries the year service matching the window to avoid cross-service duplication
    — each service spans multiple years so we only query the most recent one that
    covers the window.
    Typical return: 130-270 per 7-day window for Louisville metro.
    """
    start = date - datetime.timedelta(days=7)
    # Use only the crime_data_2026 service — it spans 2020-present
    # and avoids triple-counting from querying overlapping year services
    svc = "crime_data_2026"
    url = f"{LOJIC_BASE}/{svc}/FeatureServer/0/query"
    where = (
        f"date_occurred >= TIMESTAMP '{start.isoformat()} 00:00:00' "
        f"AND date_occurred < TIMESTAMP '{date.isoformat()} 00:00:00' "
        f"AND ({VIOLENT_OFFENSES})"
    )
    try:
        r = requests.get(url, params={
            "where": where, "returnCountOnly": True, "f": "json"
        }, timeout=10)
        r.raise_for_status()
        return {"crime_violent_7d": r.json().get("count", 0)}
    except Exception as e:
        print(f"  [WARN] LOJIC crime: {e}", file=sys.stderr)
        return {"crime_violent_7d": None}


# ── LOJIC: ROW Permits (road disruption proxy) ────────────────────────────────
# Dedicated crash service is access-restricted. ROW permits are the
# actionable signal anyway — active road closures near the hospital
# elevate MVA risk and affect EMS response times.
ROW_CATCHMENT_ZIPS = {
    "40202","40203","40204","40205","40206","40207","40208",
    "40209","40210","40211","40212","40213","40214","40215",
    "40216","40217","40218","40219","40220","40223",
}

def fetch_row_permits(date: datetime.date) -> dict:
    """Active ROW construction permits in ED catchment area."""
    url = f"{LOJIC_BASE}/Louisville_KY_ROW_Construction_Permits_new/FeatureServer/0/query"
    r = requests.get(url, params={
        "where": "1=1",
        "outFields": "ZIP",
        "resultRecordCount": 2000,
        "f": "json"
    }, timeout=15)
    r.raise_for_status()
    feats = r.json().get("features", [])
    catchment = sum(
        1 for f in feats
        if str(f.get("attributes", {}).get("ZIP", "")) in ROW_CATCHMENT_ZIPS
    )
    return {"row_permits_catchment": catchment}


# ── CDC NSSP (ER visit % + velocity) ─────────────────────────────────────────
def fetch_nssp(date: datetime.date) -> dict:
    """Fetch flu/COVID/RSV % + velocity (week-over-week change) for Kentucky."""
    r = requests.get(
        "https://data.cdc.gov/resource/rdmq-nq56.json"
        "?geography=Kentucky&$order=week_end DESC&$limit=8",
        timeout=15
    )
    r.raise_for_status()
    rows = r.json()
    result = {
        "nssp_flu_pct": None, "nssp_covid_pct": None,
        "nssp_rsv_pct": None, "nssp_flu_trend": None,
        "nssp_flu_velocity": None,
    }
    if not rows:
        return result
    week_data = {}
    for row in rows:
        wk = row.get("week_end", "")[:10]
        if wk and wk not in week_data:
            flu = row.get("percent_visits_influenza")
            cov = row.get("percent_visits_covid")
            rsv = row.get("percent_visits_rsv")
            trend = row.get("ed_trends_influenza")
            if flu is not None:
                week_data[wk] = {
                    "flu": float(flu), "cov": float(cov or 0),
                    "rsv": float(rsv or 0), "trend": trend,
                }
    weeks = sorted(week_data.keys(), reverse=True)
    if weeks:
        latest = week_data[weeks[0]]
        result["nssp_flu_pct"]   = round(latest["flu"], 2)
        result["nssp_covid_pct"] = round(latest["cov"], 2)
        result["nssp_rsv_pct"]   = round(latest["rsv"], 2)
        result["nssp_flu_trend"] = latest["trend"]
        if len(weeks) >= 2:
            prev = week_data[weeks[1]]
            result["nssp_flu_velocity"] = round(latest["flu"] - prev["flu"], 3)
    return result


# ── CDC NWSS ──────────────────────────────────────────────────────────────────
def fetch_nwss(date: datetime.date) -> dict:
    """Jefferson County wastewater COVID percentile.
    Fetches and filters client-side (server-side $where with AND is unreliable).
    NWSS data only available through ~Sep 2025 for Jefferson County."""
    ds = (date - datetime.timedelta(days=60)).isoformat()
    r = requests.get(
        "https://data.cdc.gov/resource/2ew6-ywp6.json"
        f"?county_fips=21111&$order=date_end DESC&$limit=30",
        timeout=15
    )
    r.raise_for_status()
    rows = r.json()
    # Filter: valid ptc_15d is 0-100, pick most recent valid
    valid = [
        row for row in rows
        if row.get("ptc_15d") is not None
        and 0 <= float(row["ptc_15d"]) <= 100
        and row.get("date_end", "") >= ds
    ]
    if valid:
        return {"nwss_percentile": float(valid[0]["ptc_15d"])}
    return {"nwss_percentile": None}



# ── NWS Severe Weather Alerts (free, no key) ─────────────────────────────────
def fetch_nws_alerts(date: datetime.date) -> dict:
    """Active NWS alerts for Louisville. Only meaningful for today/yesterday."""
    days_ago = (datetime.date.today() - date).days
    if days_ago > 1:
        return {"nws_alert_count": 0, "power_risk_flag": 0}
    r = requests.get(
        f"https://api.weather.gov/alerts/active?point={LAT},{LON}",
        headers={"User-Agent": "(traumayellow.com, evan.kuhl@gmail.com)"},
        timeout=10
    )
    r.raise_for_status()
    features = r.json().get("features", [])
    events = [f["properties"].get("event", "").upper() for f in features]
    power_keywords = ["WIND", "ICE", "WINTER STORM", "TORNADO", "THUNDERSTORM", "BLIZZARD"]
    power_risk = int(any(kw in ev for ev in events for kw in power_keywords))
    return {"nws_alert_count": len(features), "power_risk_flag": power_risk}


# ── School calendar (JCPS) ────────────────────────────────────────────────────
SCHOOL_OUT_RANGES = [
    ("2024-06-05", "2024-08-06"), ("2024-10-07", "2024-10-11"),
    ("2024-11-25", "2024-11-29"), ("2024-12-23", "2025-01-03"),
    ("2025-03-31", "2025-04-04"), ("2025-06-04", "2025-08-05"),
    ("2025-10-06", "2025-10-10"), ("2025-11-24", "2025-11-28"),
    ("2025-12-22", "2026-01-02"), ("2026-03-30", "2026-04-03"),
    ("2026-06-03", "2026-08-04"),
    ("2024-05-03", "2024-05-03"), ("2025-05-02", "2025-05-02"),
    ("2026-05-01", "2026-05-01"),
]

def is_school_day_out(date: datetime.date) -> bool:
    if date.weekday() >= 5:
        return True
    if date in HOLIDAYS:
        return True
    ds = date.isoformat()
    return any(s <= ds <= e for s, e in SCHOOL_OUT_RANGES)


# ── Ticketmaster ───────────────────────────────────────────────────────────────
def fetch_events(date: datetime.date) -> dict:
    tm_key = os.environ.get("TICKETMASTER_API_KEY", "")
    if not tm_key:
        return {"event_attendance": None}
    ds = date.strftime("%Y-%m-%dT00:00:00Z")
    de = date.strftime("%Y-%m-%dT23:59:59Z")
    r = requests.get(
        f"https://app.ticketmaster.com/discovery/v2/events.json"
        f"?apikey={tm_key}&city=Louisville&stateCode=KY"
        f"&startDateTime={ds}&endDateTime={de}&size=50",
        timeout=15
    )
    r.raise_for_status()
    events = r.json().get("_embedded", {}).get("events", [])
    total = 0
    for ev in events:
        venue = ev.get("_embedded", {}).get("venues", [{}])[0]
        try:
            total += int(venue.get("upcomingEvents", {}).get("_total", 0))
        except Exception:
            pass
    return {"event_attendance": total or None}


# ── Holidays ───────────────────────────────────────────────────────────────────
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
    # Kentucky Derby week — ED volume suppressed on race day, elevated Fri (Oaks)
    # Derby = first Saturday in May; Oaks = Friday before
    datetime.date(2024, 5, 3),   # Derby 2024
    datetime.date(2024, 5, 4),   # Oaks 2024 (day after, residual)
    datetime.date(2024, 4, 26),  # Thunder Over Louisville 2024
    datetime.date(2025, 5, 2),   # Derby 2025
    datetime.date(2025, 5, 3),   # Oaks 2025
    datetime.date(2025, 4, 26),  # Thunder Over Louisville 2025
    datetime.date(2026, 5, 1),   # Derby 2026
    datetime.date(2026, 5, 2),   # Oaks 2026
    datetime.date(2026, 4, 25),  # Thunder Over Louisville 2026
}


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Target date YYYY-MM-DD (default: yesterday)")
    args = parser.parse_args()

    target = datetime.date.fromisoformat(args.date) if args.date \
             else datetime.date.today() - datetime.timedelta(days=1)

    print(f"traumayellow · collect_signals.py · {target}")

    # Duplicate check
    if SIGNALS.exists():
        with open(SIGNALS) as f:
            existing = {row["date"] for row in csv.DictReader(f)}
        if target.isoformat() in existing:
            print(f"  Already have {target} — skipping.")
            return

    row = {
        "date":        target.isoformat(),
        "is_holiday":  int(target in HOLIDAYS),
        "day_of_week": target.weekday(),
    }

    weather = safe_get(lambda: fetch_weather(target), "weather")
    row.update(weather or {})

    if weather and weather.get("temp_max_f") is not None:
        delta = safe_get(lambda: compute_temp_delta(target, weather["temp_max_f"]), "temp_delta")
        row.update(delta or {})

    aq = safe_get(lambda: fetch_air_quality(target), "Open-Meteo AQ")
    row.update(aq or {})

    # AirNow current obs only if collecting for today/yesterday
    days_ago = (datetime.date.today() - target).days
    if days_ago <= 1:
        airnow = safe_get(fetch_airnow_current, "AirNow current")
        row.update(airnow or {})

    o3_lags = safe_get(lambda: fetch_ozone_lags_from_omaq(target), "O3 lags")
    row.update(o3_lags or {})

    nssp = safe_get(lambda: fetch_nssp(target), "NSSP")
    row.update(nssp or {})

    nwss = safe_get(lambda: fetch_nwss(target), "NWSS")
    row.update(nwss or {})

    crime = safe_get(lambda: fetch_lojic_crime(target), "LOJIC crime")
    row.update(crime or {})

    row_permits = safe_get(lambda: fetch_row_permits(target), "ROW permits")
    row.update(row_permits or {})

    nws = safe_get(lambda: fetch_nws_alerts(target), "NWS alerts")
    row.update(nws or {})

    # School calendar (pure date logic, no API)
    row["is_school_out"]   = int(is_school_day_out(target))
    row["is_school_night"] = int(
        not is_school_day_out(target) and target.weekday() in (6, 0, 1, 2, 3)
    )  # Sun-Thu during school weeks

    events = safe_get(lambda: fetch_events(target), "Ticketmaster")
    row.update(events or {})

    for col in COLUMNS:
        row.setdefault(col, "")

    SIGNALS.parent.mkdir(exist_ok=True)
    write_header = not SIGNALS.exists()
    with open(SIGNALS, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"  weather: {row.get('temp_max_f')}°F  delta: {row.get('temp_delta')}°F  "
          f"surge: {row.get('temp_surge_flag')}  pm25: {row.get('aqi_pm25')}  "
          f"crime_7d: {row.get('crime_violent_7d')}  row: {row.get('row_permits_catchment')}")
    print("Done.")


if __name__ == "__main__":
    main()
