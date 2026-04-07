#!/usr/bin/env python3
"""
traumayellow · send_prediction_email.py
Fetches predictions.json from GitHub Pages and sends a formatted HTML
forecast email with Z-score surge alerting.

Runs daily at 6 AM ET via GitHub Actions.

Required GitHub Secrets:
  GMAIL_APP_PW   - 16-char Gmail app password (no spaces)
  RECIPIENT      - destination email address
"""

import os
import json
import smtplib
import ssl
import datetime
import sys
import urllib.request
from email.message import EmailMessage
from email.policy import SMTP

GMAIL_USER      = "evan.kuhl@gmail.com"
GMAIL_APP_PW    = os.environ["GMAIL_APP_PW"]
RECIPIENT       = os.environ.get("RECIPIENT", "evan.kuhl@uoflhealth.org")
PREDICTIONS_URL = "https://kardioversion1.github.io/TraumaYellow_Email/predictions.json"


def fetch_predictions() -> dict:
    req = urllib.request.Request(
        PREDICTIONS_URL,
        headers={"User-Agent": "traumayellow-email/1.0"}
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def volume_tier(visits: int):
    if visits >= 70:
        return "#c0392b", "HIGH"
    elif visits >= 60:
        return "#e67e22", "MODERATE-HIGH"
    elif visits >= 50:
        return "#f39c12", "MODERATE"
    else:
        return "#27ae60", "LOW"


def alert_style(alert: str):
    if alert == "RED":
        return "#c0392b", "🔴 RED ALERT"
    elif alert == "YELLOW":
        return "#e67e22", "⚡ YELLOW ALERT"
    return None, None


def build_html(payload: dict, generated_at: str, is_stale: bool = False) -> str:
    # Filter to future dates only, take next 3
    today = datetime.date.today()
    all_forecast = payload.get("forecast", [])
    forecast = [p for p in all_forecast
                if datetime.date.fromisoformat(p["date"]) > today][:3]
    # Fallback: if all dates are past (stale predictions.json), use last 3
    if not forecast:
        forecast = all_forecast[-3:]

    # Stale data check — warn if forecast doesn't reach tomorrow
    first_forecast_date = datetime.date.fromisoformat(forecast[0]["date"]) if forecast else None
    is_stale = (first_forecast_date is None) or (first_forecast_date > today + datetime.timedelta(days=3))
    model_ver     = payload.get("model_version", "unknown")
    last_actual   = payload.get("last_actual", "unknown")
    mae           = payload.get("mae", 9)
    training_rows = payload.get("training_rows", "?")
    briefing      = payload.get("status_briefing", "Jewish Hospital ED · Daily Forecast")
    top_alert     = payload.get("top_alert")

    # Alert banner (shown when any of next 3 days is flagged)
    alert_banner = ""
    if top_alert:
        a_color, a_label = alert_style(top_alert)
        alert_banner = f"""
        <tr>
          <td colspan="4" style="background:{a_color}; padding:12px 28px; text-align:center;">
            <span style="color:#fff; font-size:15px; font-weight:700; letter-spacing:1px;">
              {a_label} — Statistically unusual volume predicted in next 3 days
            </span>
          </td>
        </tr>"""

    stale_banner = ""
    if is_stale:
        stale_banner = """
        <tr>
          <td colspan="4" style="background:#7f1d1d;padding:10px 28px;text-align:center;">
            <span style="color:#fca5a5;font-size:13px;font-weight:700;">
              &#9888; WARNING — Forecast data may be stale. Verify before staffing decisions.
            </span>
          </td>
        </tr>"""

    rows = ""
    for p in forecast:
        visits = p["predicted"]
        lo     = p.get("band_low",  max(0, visits - int(mae)))
        hi     = p.get("band_high", visits + int(mae))
        z      = p.get("z_score", 0)
        dow_m  = p.get("dow_mean", "?")
        alert  = p.get("alert")
        color, tier = volume_tier(visits)
        label  = f"{p['day']} {p['date']}"

        # Z-score badge
        z_color = "#c0392b" if z >= 2.33 else ("#e67e22" if z >= 1.65 else "#888")
        z_badge = f'<span style="font-size:10px; color:{z_color}; font-weight:600;">Z={z:+.2f}</span>'

        # Alert indicator on row
        row_bg = f"background:#fff8f0;" if alert == "YELLOW" else ("background:#fff5f5;" if alert == "RED" else "")

        rows += f"""
        <tr style="{row_bg}">
          <td style="padding:14px 18px; font-size:15px; font-weight:600;
                     border-bottom:1px solid #f0f0f0; white-space:nowrap;">{label}</td>
          <td style="padding:14px 18px; text-align:center; border-bottom:1px solid #f0f0f0;">
            <span style="font-size:30px; font-weight:700; color:{color};">{visits}</span>
            <br><span style="font-size:11px; color:#999;">({lo}&ndash;{hi} range)</span>
            <br>{z_badge}
          </td>
          <td style="padding:14px 18px; text-align:center; border-bottom:1px solid #f0f0f0;">
            <span style="display:inline-block; background:{color}; color:#fff;
                         font-size:11px; font-weight:700; padding:3px 9px;
                         border-radius:3px; letter-spacing:.5px;">{tier}</span>
          </td>
          <td style="padding:14px 18px; font-size:12px; color:#888;
                     border-bottom:1px solid #f0f0f0;">
            DOW avg<br><strong>{dow_m}</strong>
          </td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>TraumaYellow Daily Forecast</title>
</head>
<body style="margin:0;padding:0;background:#f5f5f5;
             font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0"
         style="background:#f5f5f5;padding:24px 0;">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0"
             style="background:#fff;border-radius:6px;overflow:hidden;
                    box-shadow:0 1px 4px rgba(0,0,0,.08);">

        <!-- Header -->
        <tr>
          <td colspan="4" style="background:#1a1a2e;padding:22px 28px;">
            <span style="color:#f5c518;font-size:20px;font-weight:700;letter-spacing:1px;">
              &#9889; TRAUMAYELLOW
            </span>
            <span style="color:#888;font-size:13px;margin-left:12px;">
              Jewish Hospital ED &middot; Daily Forecast
            </span>
          </td>
        </tr>

        <!-- Alert banner (conditional) -->
        {alert_banner}
        <!-- Stale data warning (conditional) -->
        {stale_banner}

        <!-- Status briefing + subheader -->
        <tr>
          <td colspan="4" style="padding:12px 28px 8px;border-bottom:2px solid #eee;">
            <div style="font-size:15px;font-weight:600;color:#1a1a2e;margin-bottom:5px;">
              {briefing}
            </div>
            <span style="font-size:12px;color:#888;">
              {generated_at} &nbsp;&middot;&nbsp; Next 3 days
              &nbsp;&middot;&nbsp; Retrained {model_ver}
              &nbsp;&middot;&nbsp; MAE &plusmn;{int(mae)}
            </span>
          </td>
        </tr>

        <!-- Table header -->
        <tr style="background:#fafafa;">
          <td style="padding:10px 18px;font-size:11px;color:#aaa;
                     text-transform:uppercase;letter-spacing:.5px;font-weight:600;">Date</td>
          <td style="padding:10px 18px;font-size:11px;color:#aaa;text-align:center;
                     text-transform:uppercase;letter-spacing:.5px;font-weight:600;">Predicted</td>
          <td style="padding:10px 18px;font-size:11px;color:#aaa;text-align:center;
                     text-transform:uppercase;letter-spacing:.5px;font-weight:600;">Volume</td>
          <td style="padding:10px 18px;font-size:11px;color:#aaa;
                     text-transform:uppercase;letter-spacing:.5px;font-weight:600;">Baseline</td>
        </tr>

        {rows}

        <!-- Z-score legend -->
        <tr>
          <td colspan="4" style="padding:12px 28px;background:#fafafa;border-top:1px solid #eee;">
            <span style="font-size:11px;color:#aaa;">
              Z-score: deviation from day-of-week historical average &nbsp;&middot;&nbsp;
              <span style="color:#e67e22;font-weight:600;">Z&ge;1.65</span> = Yellow Alert (95th pct)
              &nbsp; <span style="color:#c0392b;font-weight:600;">Z&ge;2.33</span> = Red Alert (99th pct)
            </span>
          </td>
        </tr>

        <!-- Footer -->
        <tr>
          <td colspan="4" style="padding:14px 28px 22px;border-top:1px solid #f0f0f0;">
            <p style="margin:0;font-size:12px;color:#bbb;line-height:1.7;">
              Last actual: {last_actual} &nbsp;&middot;&nbsp;
              <span style="color:#27ae60;font-weight:600;">Low</span> &lt;50 &nbsp;
              <span style="color:#f39c12;font-weight:600;">Moderate</span> 50&ndash;59 &nbsp;
              <span style="color:#e67e22;font-weight:600;">Mod-High</span> 60&ndash;69 &nbsp;
              <span style="color:#c0392b;font-weight:600;">High</span> 70+
            </p>
            <p style="margin:6px 0 0;font-size:12px;color:#bbb;">
              <a href="https://traumayellow.com"
                 style="color:#f5c518;text-decoration:none;">traumayellow.com</a>
              &nbsp;&middot;&nbsp; XGBoost &nbsp;&middot;&nbsp; Jul 2024&ndash;present
            </p>
          </td>
        </tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""


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
    print(f"  Sent → {RECIPIENT}")


def main():
    is_stale = False
    print("Fetching predictions.json...")
    payload = fetch_predictions()

    forecast = payload.get("forecast", [])
    if not forecast:
        print("No forecast data — aborting", file=sys.stderr)
        sys.exit(1)

    today_date = datetime.date.today()
    all_fc     = payload.get("forecast", [])
    future_fc  = [p for p in all_fc if datetime.date.fromisoformat(p["date"]) > today_date]
    forecast   = future_fc if future_fc else all_fc[-3:]
    p0         = forecast[0]
    today_str  = datetime.date.today().strftime("%A, %B %-d, %Y")
    now_str    = datetime.datetime.now().strftime("%b %-d %Y %I:%M %p ET")
    top_alert  = payload.get("top_alert")

    # Alert prefix in subject
    alert_prefix = ""
    if top_alert == "RED":
        alert_prefix = "🔴 RED ALERT · "
    elif top_alert == "YELLOW":
        alert_prefix = "⚡ YELLOW · "

    stale_prefix = "⚠️ STALE DATA · " if is_stale else ""
    subject = (
        f"{stale_prefix}{alert_prefix}ED Forecast {today_str}: "
        f"{p0['day']} → {p0['predicted']} visits"
    )

    is_stale = (first_forecast_date is None) or                (first_forecast_date > today + datetime.timedelta(days=3))
    print(f"Building email (top_alert={top_alert}, stale={is_stale})...")
    html = build_html(payload, now_str, is_stale=is_stale)

    print("Sending...")
    send_email(subject, html)
    print("Done.")


if __name__ == "__main__":
    main()
