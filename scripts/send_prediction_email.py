#!/usr/bin/env python3
"""
traumayellow · send_prediction_email.py
Fetches predictions.json from GitHub Pages and sends a formatted HTML
forecast email. No model loading — all prediction logic lives in retrain.py.

Runs daily at 6 AM ET via GitHub Actions.

Required GitHub Secrets:
  GMAIL_APP_PW   - 16-char Gmail app password (no spaces)
  RECIPIENT      - destination email address

Hardcoded:
  GMAIL_USER     - evan.kuhl@gmail.com
  PREDICTIONS_URL - GitHub Pages predictions.json URL
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

# ── Config ────────────────────────────────────────────────────────────────────
GMAIL_USER       = "evan.kuhl@gmail.com"
GMAIL_APP_PW     = os.environ["GMAIL_APP_PW"]
RECIPIENT        = os.environ.get("RECIPIENT", "evan.kuhl@uoflhealth.org")
PREDICTIONS_URL  = "https://kardioversion1.github.io/TraumaYellow_Email/predictions.json"
MODEL_MAE        = 9   # fallback if not in payload


# ── Fetch predictions ─────────────────────────────────────────────────────────
def fetch_predictions() -> dict:
    req = urllib.request.Request(
        PREDICTIONS_URL,
        headers={"User-Agent": "traumayellow-email/1.0"}
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


# ── Rendering ─────────────────────────────────────────────────────────────────
def volume_tier(visits: int):
    if visits >= 70:
        return "#c0392b", "HIGH"
    elif visits >= 60:
        return "#e67e22", "MODERATE-HIGH"
    elif visits >= 50:
        return "#f39c12", "MODERATE"
    else:
        return "#27ae60", "LOW"


def build_html(payload: dict, generated_at: str) -> str:
    forecast     = payload.get("forecast", [])[:3]   # next 3 days only
    model_ver    = payload.get("model_version", "unknown")
    last_actual  = payload.get("last_actual", "unknown")
    mae          = payload.get("mae", MODEL_MAE)
    training_rows= payload.get("training_rows", "?")

    rows = ""
    for p in forecast:
        visits = p["predicted"]
        lo     = p.get("band_low",  max(0, visits - int(mae)))
        hi     = p.get("band_high", visits + int(mae))
        color, tier = volume_tier(visits)
        label  = f"{p['day']} {p['date']}"

        rows += f"""
        <tr>
          <td style="padding:14px 18px; font-size:15px; font-weight:600;
                     border-bottom:1px solid #f0f0f0; white-space:nowrap;">{label}</td>
          <td style="padding:14px 18px; text-align:center; border-bottom:1px solid #f0f0f0;">
            <span style="font-size:30px; font-weight:700; color:{color};">{visits}</span>
            <br><span style="font-size:11px; color:#999;">({lo}&ndash;{hi} range)</span>
          </td>
          <td style="padding:14px 18px; text-align:center; border-bottom:1px solid #f0f0f0;">
            <span style="display:inline-block; background:{color}; color:#fff;
                         font-size:11px; font-weight:700; padding:3px 9px;
                         border-radius:3px; letter-spacing:.5px;">{tier}</span>
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
      <table width="580" cellpadding="0" cellspacing="0"
             style="background:#fff;border-radius:6px;overflow:hidden;
                    box-shadow:0 1px 4px rgba(0,0,0,.08);">

        <!-- Header -->
        <tr>
          <td colspan="3" style="background:#1a1a2e;padding:22px 28px;">
            <span style="color:#f5c518;font-size:20px;font-weight:700;letter-spacing:1px;">
              &#9889; TRAUMAYELLOW
            </span>
            <span style="color:#888;font-size:13px;margin-left:12px;">
              Jewish Hospital ED &middot; Daily Forecast
            </span>
          </td>
        </tr>

        <!-- Subheader -->
        <tr>
          <td colspan="3" style="padding:14px 28px 10px;border-bottom:2px solid #eee;">
            <span style="font-size:13px;color:#888;">
              {generated_at} &nbsp;&middot;&nbsp; Next 3 days
              &nbsp;&middot;&nbsp; Model retrained {model_ver}
              &nbsp;&middot;&nbsp; {training_rows} training rows
            </span>
          </td>
        </tr>

        <!-- Table header -->
        <tr style="background:#fafafa;">
          <td style="padding:10px 18px;font-size:11px;color:#aaa;
                     text-transform:uppercase;letter-spacing:.5px;font-weight:600;">Date</td>
          <td style="padding:10px 18px;font-size:11px;color:#aaa;text-align:center;
                     text-transform:uppercase;letter-spacing:.5px;font-weight:600;">Predicted Visits</td>
          <td style="padding:10px 18px;font-size:11px;color:#aaa;text-align:center;
                     text-transform:uppercase;letter-spacing:.5px;font-weight:600;">Volume</td>
        </tr>

        <!-- Rows -->
        {rows}

        <!-- Footer -->
        <tr>
          <td colspan="3" style="padding:16px 28px 22px;border-top:1px solid #f0f0f0;">
            <p style="margin:0;font-size:12px;color:#bbb;line-height:1.7;">
              Confidence range &plusmn;{int(mae)} visits (MAE) &nbsp;&middot;&nbsp;
              Last actual: {last_actual} &nbsp;&middot;&nbsp;
              <span style="color:#27ae60;font-weight:600;">Low</span> &lt;50 &nbsp;
              <span style="color:#f39c12;font-weight:600;">Moderate</span> 50&ndash;59 &nbsp;
              <span style="color:#e67e22;font-weight:600;">Mod-High</span> 60&ndash;69 &nbsp;
              <span style="color:#c0392b;font-weight:600;">High</span> 70+
            </p>
            <p style="margin:6px 0 0;font-size:12px;color:#bbb;">
              <a href="https://traumayellow.com"
                 style="color:#f5c518;text-decoration:none;">traumayellow.com</a>
              &nbsp;&middot;&nbsp; XGBoost &nbsp;&middot;&nbsp;
              Training data Jul 2024&ndash;present
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
    print(f"  Sent → {RECIPIENT}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Fetching predictions.json...")
    payload = fetch_predictions()

    forecast = payload.get("forecast", [])
    if not forecast:
        print("No forecast data in payload — aborting", file=sys.stderr)
        sys.exit(1)

    p0        = forecast[0]
    today_str = datetime.date.today().strftime("%A, %B %-d, %Y")
    now_str   = datetime.datetime.now().strftime("%b %-d %Y %I:%M %p ET")
    subject   = (
        f"ED Forecast {today_str}: "
        f"{p0['day']} → {p0['predicted']} visits predicted"
    )

    print("Building email...")
    html = build_html(payload, now_str)

    print("Sending...")
    send_email(subject, html)
    print("Done.")


if __name__ == "__main__":
    main()
