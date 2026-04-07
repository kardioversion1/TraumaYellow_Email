#!/usr/bin/env python3
"""
traumayellow · refresh_forecast.py
Runs daily at 5:45 AM ET to regenerate predictions.json from the
existing trained model — keeping the forecast window current without
a full weekly retrain.

Loads model.pkl, generates 14 days of predictions from today forward,
writes updated predictions.json. MAE and feature importance carry
over from the last retrain.
"""

import json, pickle, datetime
import numpy as np
import pandas as pd
from pathlib import Path

ROOT       = Path(__file__).parent.parent
MODEL_PKL  = ROOT / "model" / "model.pkl"
FEAT_JSON  = ROOT / "model" / "feature_list.json"
PREDS_JSON = ROOT / "predictions" / "predictions.json"
ED_COUNTS  = ROOT / "data" / "ed_counts.csv"
METRICS    = ROOT / "data" / "model_metrics.json"

def main():
    print(f"traumayellow · refresh_forecast.py · {datetime.datetime.utcnow().isoformat()} UTC")

    if not MODEL_PKL.exists():
        print("  model.pkl not found — skipping (run retrain first)")
        return

    # Load model
    with open(MODEL_PKL, "rb") as f:
        bundle = pickle.load(f)
    model        = bundle["model"]
    feature_cols = bundle["feature_cols"]
    epoch        = pd.Timestamp(bundle["epoch"])
    print(f"  Model loaded (epoch={bundle['epoch']}, features={len(feature_cols)})")

    # Load existing predictions.json to carry over MAE, baselines, etc.
    existing = {}
    if PREDS_JSON.exists():
        with open(PREDS_JSON) as f:
            existing = json.load(f)

    mae          = existing.get("mae", 9)
    dow_baselines = existing.get("dow_baselines", {})
    top_features = existing.get("top_features", {})
    status_briefing = existing.get("status_briefing", "ED Forecast")

    # Load ed_counts for autoregressive features
    ed = pd.read_csv(ED_COUNTS, parse_dates=["date"]).sort_values("date")
    last_known = ed.set_index("date")["total_visits"]

    # Generate 14 days from today
    today = datetime.date.today()
    predictions = []
    for i in range(1, 15):
        td = pd.Timestamp(today) + pd.Timedelta(days=i)
        days_since_epoch = (td - epoch).days
        month_index = (td.year - epoch.year) * 12 + (td.month - epoch.month)
        dow = td.dayofweek

        visits_lag7   = float(last_known.iloc[-7])   if len(last_known) >= 7  else np.nan
        visits_lag14  = float(last_known.iloc[-14])  if len(last_known) >= 14 else np.nan
        visits_roll7  = float(np.mean(last_known.iloc[-7:]))  if len(last_known) >= 7  else np.nan
        visits_roll14 = float(np.mean(last_known.iloc[-14:])) if len(last_known) >= 14 else np.nan
        visits_roll28 = float(np.mean(last_known.iloc[-28:])) if len(last_known) >= 28 else np.nan

        base = {
            "days_since_epoch": days_since_epoch,
            "month_index":      month_index,
            "dow_sin":          np.sin(2 * np.pi * dow / 7),
            "dow_cos":          np.cos(2 * np.pi * dow / 7),
            "month_sin":        np.sin(2 * np.pi * td.month / 12),
            "month_cos":        np.cos(2 * np.pi * td.month / 12),
            "day_of_week":      dow,
            "is_weekend":       int(dow >= 5),
            "is_monday":        int(dow == 0),
            "is_friday":        int(dow == 4),
            "month":            td.month,
            "day_of_month":     td.day,
            "is_holiday":       0,
            "temp_delta":       0.0,
            "temp_surge_flag":  0,
            "aqi_o3_wma":       0.0,
            "visits_lag7":      visits_lag7,
            "visits_lag14":     visits_lag14,
            "visits_roll7":     visits_roll7,
            "visits_roll14":    visits_roll14,
            "visits_roll28":    visits_roll28,
        }

        row = pd.DataFrame([base])
        for col in feature_cols:
            if col not in row.columns:
                row[col] = 0
        row = row[feature_cols].fillna(0)

        pred = float(np.maximum(model.predict(row)[0], 20))
        pred_rounded = round(pred)

        dow_key  = str(dow)
        dow_mean = dow_baselines.get(dow_key, {}).get("mean", pred)
        dow_std  = dow_baselines.get(dow_key, {}).get("std", 10)
        z_score  = round((pred - dow_mean) / dow_std, 2) if dow_std > 0 else 0.0

        alert = "RED" if z_score >= 2.33 else ("YELLOW" if z_score >= 1.65 else None)

        predictions.append({
            "date":      td.strftime("%Y-%m-%d"),
            "day":       td.strftime("%A"),
            "predicted": pred_rounded,
            "band_low":  max(0, round(pred - mae)),
            "band_high": round(pred + mae),
            "z_score":   z_score,
            "dow_mean":  round(dow_mean, 1),
            "alert":     alert,
        })

    alerts_3day = [p["alert"] for p in predictions[:3] if p["alert"]]
    top_alert = "RED" if "RED" in alerts_3day else ("YELLOW" if "YELLOW" in alerts_3day else None)

    payload = {
        "generated_at":    datetime.datetime.utcnow().isoformat() + "Z",
        "model_version":   existing.get("model_version", datetime.date.today().isoformat()),
        "training_rows":   existing.get("training_rows", "?"),
        "holdout_rows":    existing.get("holdout_rows", "?"),
        "epoch":           str(epoch.date()),
        "last_actual":     str(ed["date"].max().date()),
        "mae":             mae,
        "metrics":         existing.get("metrics", {}),
        "top_features":    top_features,
        "dow_baselines":   dow_baselines,
        "top_alert":       top_alert,
        "status_briefing": status_briefing,
        "forecast":        predictions,
    }

    PREDS_JSON.parent.mkdir(exist_ok=True)
    with open(PREDS_JSON, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"  predictions.json refreshed — {len(predictions)} days from {predictions[0]['date']} to {predictions[-1]['date']}")
    print(f"  top_alert={top_alert}  MAE={mae}  briefing='{status_briefing}'")
    print("Done.")

if __name__ == "__main__":
    main()
