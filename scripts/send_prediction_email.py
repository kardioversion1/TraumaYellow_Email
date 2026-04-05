"""
send_prediction_email.py
Generates 1/2/3-day ED visit predictions and emails them to the recipient.

Required GitHub Secrets:
  GMAIL_USER    - sender Gmail address (e.g. traumayellow.alerts@gmail.com)
  GMAIL_APP_PW  - 16-char Gmail app password (no spaces)
  RECIPIENT     - destination address (evan.kuhl@uoflhealth.org)

Model artifacts expected at:
  model/xgb_model.json      - trained XGBoost model
  model/feature_list.json   - ordered list of feature names
  data/daily_totals.csv     - rolling dataset with columns: date, total_visits
"""

import os
import json
import smtplib
import ssl
import datetime
import sys
from email.message import EmailMessage
from email.policy import SMTP
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xgboost as xgb

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH    = Path("model/xgb_model.json")
FEATURES_PATH = Path("model/feature_list.json")
DATA_PATH     = Path("data/daily_totals.csv")

# ── Credentials from environment ───────────────────────────────────────────────
GMAIL_USER   = os.environ.get("GMAIL_USER", "evan.kuhl@gmail.com")
GMAIL_APP_PW = os.environ["GMAIL_APP_PW"]
RECIPIENT    = os.environ.get("RECIPIENT", "evan.kuhl@uoflhealth.org")

# Louisville lat/lon for weather
LAT, LON = 38.2527, -85.7585

# Model MAE (v3 baseline) used for confidence band
MODEL_MAE = 9

# Federal holidays 2026 (expand each year)
FEDERAL_HOLIDAYS = {
    datetime.date(2026, 1, 1),
    datetime.date(2026, 1, 19),
    datetime.date(2026, 2, 16),
    datetime.date(2026, 5, 25),
    datetime.date(2026, 7, 3),
    datetime.date(2026, 9, 7),
    datetime.date(2026, 11, 11),
    datetime.date(2026, 11, 26),
    datetime.date(2026, 12, 25),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model():
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    with open(FEATURES_PATH) as f:
        feature_names = json.load(f)
    return model, feature_names


def load_history() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_weather(target_date: datetime.date) -> dict:
    """Open-Meteo forecast (free, no API key)."""
    ds = target_date.isoformat()
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={LAT}&longitude={LON}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
            f"&temperature_unit=fahrenheit"
            f"&start_date={ds}&end_date={ds}"
            f"&timezone=America%2FNew_York"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        d = r.json()["daily"]
        return {
            "temp_max_f": d["temperature_2m_max"][0],
            "temp_min_f": d["temperature_2m_min"][0],
            "precip_mm":  d["precipitation_sum"][0] or 0.0,
        }
    except Exception as e:
        print(f"  [weather] {target_date}: {e}", file=sys.stderr)
        return {"temp_max_f": None, "temp_min_f": None, "precip_mm": None}


def is_holiday(d: datetime.date) -> bool:
    return d in FEDERAL_HOLIDAYS


def is_holiday_adjacent(d: datetime.date) -> bool:
    return is_holiday(d - datetime.timedelta(days=1)) or is_holiday(d + datetime.timedelta(days=1))


def build_feature_row(target_date: datetime.date, hist_df: pd.DataFrame, feature_names: list):
    td = target_date
    epoch = datetime.date(2024, 7, 1)
    days_since_epoch = (td - epoch).days
    month_index = (td.year - 2024) * 12 + td.month
    dow = td.weekday()  # 0=Mon...6=Sun

    past = hist_df[hist_df["date"] < pd.Timestamp(td)]
    visits_7d  = past.tail(7)["total_visits"].mean()  if len(past) >= 7  else past["total_visits"].mean()
    visits_14d = past.tail(14)["total_visits"].mean() if len(past) >= 14 else past["total_visits"].mean()
    visits_28d = past.tail(28)["total_visits"].mean() if len(past) >= 28 else past["total_visits"].mean()

    w = fetch_weather(td)

    base = {
        "day_of_week":       dow,
        "month":             td.month,
        "day_of_month":      td.day,
        "is_weekend":        int(dow >= 5),
        "is_monday":         int(dow == 0),
        "is_friday":         int(dow == 4),
        "days_since_epoch":  days_since_epoch,
        "month_index":       month_index,
        "visits_7d_avg":     visits_7d,
        "visits_14d_avg":    visits_14d,
        "visits_28d_avg":    visits_28d,
        "temp_max_f":        w["temp_max_f"] if w["temp_max_f"] is not None else 65.0,
        "temp_min_f":        w["temp_min_f"] if w["temp_min_f"] is not None else 45.0,
        "precip_mm":         w["precip_mm"]  if w["precip_mm"]  is not None else 0.0,
        "is_holiday":        int(is_holiday(td)),
        "is_holiday_adj":    int(is_holiday_adjacent(td)),
    }

    row = pd.DataFrame([base])
    # Align to exact training feature list; fill unknown cols with 0
    for col in feature_names:
        if col not in row.columns:
            row[col] = 0
    return row[feature_names], w


def predict(model, feature_names, hist_df) -> list:
    today = datetime.date.today()
    results = []
    for offset in range(1, 4):
        target = today + datetime.timedelta(days=offset)
        row, weather = build_feature_row(target, hist_df, feature_names)
        raw = float(model.predict(row)[0])
        visits = max(20, round(raw))
        print(f"  {target}: raw={raw:.1f} -> {visits} visits")
        results.append({
            "date":    target,
            "label":   target.strftime("%A %b %-d"),
            "visits":  visits,
            "weather": weather,
        })
    return results


# ── Email rendering ────────────────────────────────────────────────────────────

def volume_tier(visits: int):
    if visits >= 70:
        return "#c0392b", "HIGH"
    elif visits >= 60:
        return "#e67e22", "MODERATE-HIGH"
    elif visits >= 50:
        return "#f39c12", "MODERATE"
    else:
        return "#27ae60", "LOW"


def fmt_weather(w: dict) -> str:
    if w.get("temp_max_f") is None:
        return "Weather unavailable"
    lo = round(w["temp_min_f"])
    hi = round(w["temp_max_f"])
    mm = w.get("precip_mm", 0) or 0
    wx = f"&#127783; {mm:.1f} mm rain" if mm > 1 else "&#9728; Dry"
    return f"{lo}&ndash;{hi}&deg;F &nbsp;&middot;&nbsp; {wx}"


def build_html(predictions: list, generated_at: str) -> str:
    rows = ""
    for p in predictions:
        color, tier = volume_tier(p["visits"])
        lo = max(0, p["visits"] - MODEL_MAE)
        hi = p["visits"] + MODEL_MAE
        wx = fmt_weather(p["weather"])
        rows += f"""
        <tr>
          <td style="padding:14px 18px; font-size:15px; font-weight:600; border-bottom:1px solid #f0f0f0; white-space:nowrap;">
            {p['label']}
          </td>
          <td style="padding:14px 18px; text-align:center; border-bottom:1px solid #f0f0f0;">
            <span style="font-size:30px; font-weight:700; color:{color};">{p['visits']}</span>
            <br><span style="font-size:11px; color:#999;">({lo}&ndash;{hi} range)</span>
          </td>
          <td style="padding:14px 18px; text-align:center; border-bottom:1px solid #f0f0f0;">
            <span style="display:inline-block; background:{color}; color:#fff; font-size:11px;
                         font-weight:700; padding:3px 9px; border-radius:3px; letter-spacing:.5px;">
              {tier}
            </span>
          </td>
          <td style="padding:14px 18px; font-size:13px; color:#666; border-bottom:1px solid #f0f0f0;">
            {wx}
          </td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TraumaYellow Daily Prediction</title>
</head>
<body style="margin:0; padding:0; background:#f5f5f5; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f5f5f5; padding:24px 0;">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0"
             style="background:#fff; border-radius:6px; overflow:hidden; box-shadow:0 1px 4px rgba(0,0,0,.08);">

        <!-- Header -->
        <tr>
          <td colspan="4" style="background:#1a1a2e; padding:22px 28px;">
            <span style="color:#f5c518; font-size:20px; font-weight:700; letter-spacing:1px;">&#9889; TRAUMAYELLOW</span>
            <span style="color:#888; font-size:13px; margin-left:12px;">Jewish Hospital ED &middot; Daily Forecast</span>
          </td>
        </tr>

        <!-- Subheader -->
        <tr>
          <td colspan="4" style="padding:14px 28px 10px; border-bottom:2px solid #eee;">
            <span style="font-size:13px; color:#888;">
              Generated {generated_at} &nbsp;&middot;&nbsp; Next 3 days
            </span>
          </td>
        </tr>

        <!-- Table header -->
        <tr style="background:#fafafa;">
          <td style="padding:10px 18px; font-size:11px; color:#aaa; text-transform:uppercase; letter-spacing:.5px; font-weight:600;">Date</td>
          <td style="padding:10px 18px; font-size:11px; color:#aaa; text-transform:uppercase; letter-spacing:.5px; font-weight:600; text-align:center;">Predicted Visits</td>
          <td style="padding:10px 18px; font-size:11px; color:#aaa; text-transform:uppercase; letter-spacing:.5px; font-weight:600; text-align:center;">Volume</td>
          <td style="padding:10px 18px; font-size:11px; color:#aaa; text-transform:uppercase; letter-spacing:.5px; font-weight:600;">Weather</td>
        </tr>

        <!-- Prediction rows -->
        {rows}

        <!-- Footer -->
        <tr>
          <td colspan="4" style="padding:16px 28px 22px; border-top:1px solid #f0f0f0;">
            <p style="margin:0; font-size:12px; color:#bbb; line-height:1.7;">
              Confidence range &plusmn;{MODEL_MAE} visits (model MAE) &nbsp;&middot;&nbsp;
              <span style="color:#27ae60; font-weight:600;">Low</span> &lt;50 &nbsp;
              <span style="color:#f39c12; font-weight:600;">Moderate</span> 50&ndash;59 &nbsp;
              <span style="color:#e67e22; font-weight:600;">Mod-High</span> 60&ndash;69 &nbsp;
              <span style="color:#c0392b; font-weight:600;">High</span> 70+
            </p>
            <p style="margin:6px 0 0; font-size:12px; color:#bbb;">
              <a href="https://traumayellow.com" style="color:#f5c518; text-decoration:none;">traumayellow.com</a>
              &nbsp;&middot;&nbsp; XGBoost v3 &nbsp;&middot;&nbsp; Training Jul 2024&ndash;Feb 2026
            </p>
          </td>
        </tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""


# ── Send ───────────────────────────────────────────────────────────────────────

def send_email(subject: str, html_body: str) -> None:
    msg = EmailMessage(policy=SMTP)
    msg["Subject"] = subject
    msg["From"]    = f"TraumaYellow <{GMAIL_USER}>"
    msg["To"]      = RECIPIENT
    msg.add_alternative(html_body, subtype="html")

    ctx = ssl.create_default_context()
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.ehlo()
        server.starttls(context=ctx)
        server.ehlo()
        server.login(GMAIL_USER, GMAIL_APP_PW)
        server.send_message(msg)
    print(f"  Sent -> {RECIPIENT}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("Loading model...")
    model, feature_names = load_model()

    print("Loading history...")
    hist_df = load_history()

    print("Generating predictions...")
    predictions = predict(model, feature_names, hist_df)

    today_str = datetime.date.today().strftime("%A, %B %-d, %Y")
    now_str   = datetime.datetime.now().strftime("%b %-d %Y %I:%M %p ET")
    p0        = predictions[0]
    subject   = f"ED Forecast {today_str}: {p0['label']} -> {p0['visits']} visits predicted"

    print("Building HTML...")
    html = build_html(predictions, now_str)

    print("Sending email...")
    send_email(subject, html)
    print("Done.")


if __name__ == "__main__":
    main()
