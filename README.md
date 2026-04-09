# traumayellow

ED patient volume prediction and staffing intelligence for Jewish Hospital Downtown (UofL Health).

> **see the surge before it arrives**

Predicts emergency department visit counts 1-3 days out using weather, air quality, epidemiological surveillance, events, crime, and traffic signals. Includes a live staffing dashboard at [traumayellow.com](https://traumayellow.com) and a daily 6 AM forecast email.

---

## Current status

| Item | Status |
|---|---|
| Training data | Jul 2024 - Apr 2026 (Jewish Downtown only) |
| Model | XGBoost · MAE 7.45 · last retrain 2026-04-07 |
| Dashboard | Live at traumayellow.com |
| Daily email | Live · 6 AM ET |
| Signal collection | Daily · 5:30 AM ET |
| Weekly retrain | Sundays · 7 AM ET |

---

## What it does

- Collects environmental, epidemiological, event, crime, and traffic signals daily via GitHub Actions
- Trains an XGBoost model weekly on all available Cerner ED visit data + external signals
- Predicts next 1-14 days of ED volume with confidence bands
- Writes a persistent predictions history CSV alongside the model
- Sends a daily 6 AM forecast email with volume tier, active flags, and weather context
- Serves a live staffing intelligence dashboard at traumayellow.com

---

## Repository structure

```
data/
  ed_counts.csv              Real Cerner ED visit totals — Jewish Downtown, Jul 2024-present
  signals.csv                Daily feature matrix (weather, AQ, epidemiology, events, crime)
  model_metrics.json         Latest retrain stats (MAE, within_10, surge_recall, DOW baselines)
  predictions_history.csv    Every model prediction ever made, with actuals alongside

model/
  model.pkl                  Trained XGBoost model (pickled dict: model, feature_cols, epoch)
  feature_list.json          Active feature column list

predictions/
  predictions.json           Forward forecast for today+1 through today+14 (GitHub Pages)

cerner_raw/
  README.md                  Naming convention and instructions for raw Cerner exports
  (raw CSV exports dropped here before each retrain)

scripts/
  collect_signals.py         Daily signal collection (weather, AQ, NSSP, NWSS, crime, permits, events)
  retrain.py                 Weekly model retrain — trains XGBoost, writes all data outputs
  refresh_forecast.py        Daily forward forecast refresh using existing model
  send_prediction_email.py   Daily 6 AM forecast email
  backfill.py                One-time historical signal backfill utility

.github/workflows/
  collect.yml                Daily 5:30 AM ET — collect signals, commit signals.csv
  refresh.yml                Daily 5:45 AM ET — refresh forward forecast, deploy to GitHub Pages
  daily_email.yml            Daily 6:00 AM ET — send forecast email
  retrain.yml                Weekly Sunday 7:00 AM ET — full retrain, commit all outputs
```

---

## Data sources

| Source | Feed | Status |
|---|---|---|
| Open-Meteo | Weather + air quality archive | Live |
| AirNow EPA | Real-time AQI supplement | Live |
| LOJIC / ArcGIS | LMPD crime incidents | Live |
| LOJIC / ArcGIS | ROW construction permits | Live |
| Ticketmaster | Louisville events calendar | Live |
| Google Maps | EMS access / traffic | Live |
| CDC NSSP | KY ED respiratory % | Feed present, column currently empty |
| CDC NWSS | National wastewater surveillance | Feed present, column currently empty |
| Cerner | ED visit totals (Jewish Downtown) | Manual export — not yet automated |

---

## GitHub Secrets required

| Secret | Used by | Purpose |
|---|---|---|
| `AIRNOW_API_KEY` | collect.yml | AirNow EPA real-time AQI |
| `TICKETMASTER_API_KEY` | collect.yml | Ticketmaster events |
| `GMAIL_APP_PW` | daily_email.yml | Gmail app password for forecast email |
| `RECIPIENT` | daily_email.yml | Forecast email destination address |

---

## Known data gaps

- **Cerner boarding / external transfer rows** not yet pulled — needed to properly model boarding-driven volume
- **Full 18-month ED Activity Log** not yet pulled (currently Jul 2024 - Mar 2026)
- **NSSP / NWSS columns empty** in signals.csv — collect script needs verification
- **Jewish Downtown only** — other UofL Health sites not yet in the model

---

## Daily workflow order

```
05:30 AM ET   collect.yml        collect_signals.py     append yesterday to signals.csv
05:45 AM ET   refresh.yml        refresh_forecast.py    regenerate predictions.json with today's signals
06:00 AM ET   daily_email.yml    send_prediction_email.py  send forecast email
Sunday 7 AM   retrain.yml        retrain.py             full retrain, update model + predictions_history.csv
```
