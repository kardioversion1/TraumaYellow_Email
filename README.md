# traumayellow

ED patient volume prediction and staffing intelligence for Jewish Hospital Downtown (UofL Health). Predicts emergency department visit counts 1–3 days out using weather, events, crime, and epidemiological signals. Includes a live dashboard at traumayellow.com and a daily forecast email.

## What it does

- Pulls environmental, epidemiological, event, crime, and traffic signals daily
- Trains an XGBoost model on historical ED visit data (Jul 2024–present)
- Predicts next 1–3 days of ED volume with confidence ranges
- Sends a daily 6 AM forecast email with volume tier and weather context
- Serves a staffing intelligence dashboard at traumayellow.com

## Stack

- Python, XGBoost, pandas
- GitHub Actions (data collection, retraining, daily email)
- Open-Meteo (weather), AirNow (air quality), Ticketmaster (events), LOJIC/ArcGIS (crime, permits)
- Hosted on StableServer, domain traumayellow.com

## Structure
scripts/          data collection, retraining, email
model/            saved XGBoost model + feature list
data/             rolling daily totals dataset
.github/workflows/ collect, retrain, daily email

## GitHub Secrets required

| Secret | Purpose |
|---|---|
| `GMAIL_APP_PW` | Gmail app password for daily forecast email |
| `RECIPIENT` | Email destination for daily forecast |
