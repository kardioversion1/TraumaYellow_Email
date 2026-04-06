#!/usr/bin/env python3
"""
traumayellow · retrain.py
Runs every Sunday at 07:00 ET via GitHub Actions.
Joins signals.csv + ed_counts.csv, trains XGBoost on all available labeled data,
evaluates on a 14-day temporal holdout, writes predictions.json + model_metrics.json,
and saves model.pkl.

Sets GitHub Actions environment variables:
  RETRAIN_STATUS   ok | skip | error
  RETRAIN_MAE      float
  RETRAIN_ROWS     int

data/ed_counts.csv expected columns:
  date, total_visits  (plus optionally: admissions, lwbs, etc.)
"""

import os
import sys
import json
import pickle
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from xgboost import XGBRegressor
except ImportError:
    print("xgboost not installed", file=sys.stderr)
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
SIGNALS    = ROOT / "data" / "signals.csv"
ED_COUNTS  = ROOT / "data" / "ed_counts.csv"
MODEL_PKL  = ROOT / "model" / "model.pkl"
FEAT_JSON  = ROOT / "model" / "feature_list.json"
PREDS_JSON = ROOT / "predictions" / "predictions.json"
METRICS    = ROOT / "data" / "model_metrics.json"

MIN_ROWS   = 60   # don't retrain until we have at least this many joined rows
HOLDOUT    = 14   # days withheld for evaluation


def gha_set(key, value):
    env_file = os.environ.get("GITHUB_ENV")
    if env_file:
        with open(env_file, "a") as f:
            f.write(f"{key}={value}\n")
    print(f"  ENV {key}={value}")


def load_and_join() -> pd.DataFrame:
    ed_counts = pd.read_csv(ED_COUNTS, parse_dates=["date"])
    ed_counts = ed_counts.sort_values("date").reset_index(drop=True)
    # Merge signals if available and non-empty
    if SIGNALS.exists():
        signals = pd.read_csv(SIGNALS, parse_dates=["date"])
        if len(signals) > 0:
            df = ed_counts.merge(signals, on="date", how="left")
            print(f"  Signals joined: {signals['date'].nunique()} signal days merged (left join)")
        else:
            df = ed_counts
            print("  signals.csv empty — training on ed_counts + engineered features only")
    else:
        df = ed_counts
        print("  No signals.csv — training on ed_counts + engineered features only")
    return df


def build_features(df: pd.DataFrame, epoch: pd.Timestamp):
    """Engineer features. Returns X (DataFrame), y (Series), feature_cols (list)."""

    df = df.copy()

    # Trend features anchored to training epoch
    df["days_since_epoch"] = (df["date"] - epoch).dt.days
    df["month_index"] = (
        (df["date"].dt.year  - epoch.year)  * 12
      + (df["date"].dt.month - epoch.month)
    )

    # Cyclical encodings
    df["dow_sin"]   = np.sin(2 * np.pi * df["date"].dt.dayofweek / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["date"].dt.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)

    # Calendar flags
    df["is_weekend"]   = (df["date"].dt.dayofweek >= 5).astype(int)
    df["is_monday"]    = (df["date"].dt.dayofweek == 0).astype(int)
    df["is_friday"]    = (df["date"].dt.dayofweek == 4).astype(int)
    df["month"]        = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day

    # Autoregressive lag features (shift to avoid leakage)
    df["visits_lag7"]   = df["total_visits"].shift(7)
    df["visits_lag14"]  = df["total_visits"].shift(14)
    df["visits_roll7"]  = df["total_visits"].shift(1).rolling(7).mean()
    df["visits_roll14"] = df["total_visits"].shift(1).rolling(14).mean()
    df["visits_roll28"] = df["total_visits"].shift(1).rolling(28).mean()

    # ── Temperature delta (behavioral surge signal) ───────────────────────────
    # Sudden warmth drives trauma/behavioral visits more than absolute temperature.
    # Computed as today's max temp minus the 7-day rolling mean of max temp.
    if "temp_max_f" in df.columns:
        df["temp_7d_mean"]     = df["temp_max_f"].rolling(7, min_periods=1).mean()
        df["temp_delta"]       = df["temp_max_f"] - df["temp_7d_mean"]
        df["temp_surge_flag"]  = (df["temp_delta"] > 15).astype(int)
    else:
        df["temp_delta"]      = 0.0
        df["temp_surge_flag"] = 0

    # ── Ozone lag WMA (respiratory lag signal) ────────────────────────────────
    # High O3 causes lung inflammation peaking in ED visits 3-5 days after exposure.
    # Weighted moving average: weight 3 on lag-3, 2 on lag-4, 1 on lag-5 (total=6).
    o3_col = "aqi_o3_ugm3" if "aqi_o3_ugm3" in df.columns else "aqi_o3"
    if o3_col in df.columns:
        o3 = df["aqi_o3"].fillna(method="ffill")
        df["aqi_o3_lag3"] = o3.shift(3)
        df["aqi_o3_lag4"] = o3.shift(4)
        df["aqi_o3_lag5"] = o3.shift(5)
        df["aqi_o3_wma"]  = (
            3 * df["aqi_o3_lag3"].fillna(0) +
            2 * df["aqi_o3_lag4"].fillna(0) +
            1 * df["aqi_o3_lag5"].fillna(0)
        ) / 6.0
    else:
        df["aqi_o3_wma"] = 0.0

    FEATURE_COLS = [
        # trend
        "days_since_epoch", "month_index",
        # cyclical
        "dow_sin", "dow_cos", "month_sin", "month_cos",
        # calendar
        "day_of_week", "is_weekend", "is_monday", "is_friday",
        "month", "day_of_month", "is_holiday",
        # weather
        "temp_max_f", "temp_min_f", "precip_mm", "snowfall_mm",
        "wind_max_mph", "cloud_cover_pct",
        # temperature delta (behavioral surge signal)
        "temp_delta", "temp_surge_flag",
        # air quality — Open-Meteo AQ (primary, full history)
        "aqi_pm25", "aqi_o3_ugm3",
        # air quality — AirNow EPA (current obs supplement)
        "aqi_o3", "aqi_pm25_airnow", "aqi_overall",
        # ozone WMA respiratory lag (3-5 day weighted avg)
        "aqi_o3_wma",
        # disease surveillance
        "nssp_flu_pct", "nssp_covid_pct", "nwss_percentile",
        # community signals (verified LOJIC endpoints)
        "crime_violent_7d", "row_permits_catchment",
        # events
        "event_attendance",
        # autoregressive
        "visits_lag7", "visits_lag14",
        "visits_roll7", "visits_roll14", "visits_roll28",
    ]

    # Only keep cols that exist in df
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].copy()
    y = df["total_visits"].copy()
    return X, y, feature_cols, df


def compute_dow_baselines(df: pd.DataFrame) -> dict:
    """
    Compute per-day-of-week mean and std from training data.
    Used downstream to calculate Z-scores for surge alerting.
    Returns dict: {0: {mean, std}, 1: {mean, std}, ..., 6: {mean, std}}
    """
    df = df.copy()
    df["dow"] = df["date"].dt.dayofweek
    baselines = {}
    for dow in range(7):
        subset = df[df["dow"] == dow]["total_visits"]
        baselines[str(dow)] = {
            "mean": round(float(subset.mean()), 2),
            "std":  round(float(subset.std()),  2),
            "n":    int(len(subset)),
        }
    return baselines


def train_model(X_train, y_train) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def evaluate(model, X_test, y_test) -> dict:
    preds = np.maximum(model.predict(X_test), 0)
    mae   = float(np.mean(np.abs(preds - y_test.values)))
    within_10 = float(np.mean(np.abs(preds - y_test.values) <= 10) * 100)
    surge_mask = y_test.values > (y_test.mean() + y_test.std())
    surge_recall = float(np.mean(preds[surge_mask] > y_test.mean()) * 100) \
        if surge_mask.any() else None
    return {
        "mae":          round(mae, 2),
        "within_10":    round(within_10, 1),
        "surge_recall": round(surge_recall, 1) if surge_recall is not None else None,
    }


def generate_predictions(model, df: pd.DataFrame, epoch, feature_cols: list,
                          dow_baselines: dict, mae: float) -> list:
    """Generate predictions covering today+1 through today+14.
    If last_actual is in the past, we extend the window so the forecast
    always contains future dates relative to when the email runs."""
    last_date  = df["date"].max()
    last_known = df.set_index("date")["total_visits"]

    # Ensure forecast always reaches at least today+14 from today, not from last_actual
    today = pd.Timestamp.now().normalize()
    # Start from last_actual+1 but generate enough days to cover today+14
    days_since_last = max(0, (today - last_date).days)
    n_days = max(14, days_since_last + 14)  # enough to cover today+14

    results = []
    for i in range(1, n_days + 1):
        td = last_date + pd.Timedelta(days=i)
        # Only include dates from today onwards (skip stale past predictions)
        if td < today:
            continue
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
            "temp_delta":       0.0,   # conservative — no future weather yet
            "temp_surge_flag":  0,
            "aqi_o3_wma":       0.0,
            "visits_lag7":      visits_lag7,
            "visits_lag14":     visits_lag14,
            "visits_roll7":     visits_roll7,
            "visits_roll14":    visits_roll14,
            "visits_roll28":    visits_roll28,
        }

        # Fill remaining signal columns from last known row
        last_row = df.iloc[-1]
        for col in feature_cols:
            if col not in base and col in last_row.index:
                base[col] = last_row[col]

        row = pd.DataFrame([base])
        for col in feature_cols:
            if col not in row.columns:
                row[col] = 0
        row = row[feature_cols].fillna(0)

        pred = float(np.maximum(model.predict(row)[0], 20))
        pred_rounded = round(pred)

        # ── Z-score surge alert ───────────────────────────────────────────────
        dow_key = str(dow)
        dow_mean = dow_baselines[dow_key]["mean"]
        dow_std  = dow_baselines[dow_key]["std"]
        z_score  = round((pred - dow_mean) / dow_std, 2) if dow_std > 0 else 0.0

        # Alert tiers: Yellow ≥ 1.65 (95th pct), Red ≥ 2.33 (99th pct)
        if z_score >= 2.33:
            alert = "RED"
        elif z_score >= 1.65:
            alert = "YELLOW"
        else:
            alert = None

        results.append({
            "date":      td.strftime("%Y-%m-%d"),
            "day":       td.strftime("%A"),
            "predicted": pred_rounded,
            "band_low":  max(0, round(pred - mae)),
            "band_high": round(pred + mae),
            "z_score":   z_score,
            "dow_mean":  round(dow_mean, 1),
            "alert":     alert,
        })
        if len(results) >= 14:
            break

    return results


def main():
    print(f"traumayellow · retrain.py · {datetime.datetime.utcnow().isoformat()} UTC")

    if not ED_COUNTS.exists():
        print("  ed_counts.csv not found — skipping")
        gha_set("RETRAIN_STATUS", "skip")
        return

    df = load_and_join()
    print(f"  Joined rows: {len(df)}  ({df.date.min().date()} → {df.date.max().date()})")

    if len(df) < MIN_ROWS:
        print(f"  Only {len(df)} rows — need {MIN_ROWS}. Skipping.")
        gha_set("RETRAIN_STATUS", "skip")
        return

    df = df.dropna(subset=["total_visits"])
    epoch = df["date"].min()

    X, y, feature_cols, df = build_features(df, epoch)

    # Drop warmup rows where lag features are NaN
    valid = X["visits_lag14"].notna()
    X, y  = X[valid].fillna(0), y[valid]
    df    = df[valid].reset_index(drop=True)
    print(f"  Training rows after warmup: {len(X)}")

    # Compute DOW baselines from full dataset (before train/test split)
    dow_baselines = compute_dow_baselines(df)
    print(f"  DOW baselines: Mon={dow_baselines['0']['mean']:.1f} Fri={dow_baselines['4']['mean']:.1f} Sat={dow_baselines['5']['mean']:.1f}")

    # Temporal split
    split = len(X) - HOLDOUT
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f"  Train: {len(X_train)}  Holdout: {len(X_test)}")

    print("  Training XGBoost...")
    model = train_model(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)
    print(f"  MAE={metrics['mae']}  within_10={metrics['within_10']}%  surge_recall={metrics['surge_recall']}%")

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
    top10 = dict(sorted(importance.items(), key=lambda x: -x[1])[:10])
    print(f"  Top features: {', '.join(top10.keys())}")

    # Save model + feature list
    MODEL_PKL.parent.mkdir(exist_ok=True)
    with open(MODEL_PKL, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols, "epoch": str(epoch.date())}, f)
    with open(FEAT_JSON, "w") as f:
        json.dump(feature_cols, f)
    print(f"  Model → {MODEL_PKL}")

    # Write predictions.json
    predictions = generate_predictions(model, df, epoch, feature_cols,
                                        dow_baselines, metrics["mae"])

    # Summary alert for the next 3 days
    alerts_3day = [p["alert"] for p in predictions[:3] if p["alert"]]
    top_alert = "RED" if "RED" in alerts_3day else ("YELLOW" if "YELLOW" in alerts_3day else None)

    PREDS_JSON.parent.mkdir(exist_ok=True)
    payload = {
        "generated_at":    datetime.datetime.utcnow().isoformat() + "Z",
        "model_version":   datetime.date.today().isoformat(),
        "training_rows":   len(X_train),
        "holdout_rows":    len(X_test),
        "epoch":           str(epoch.date()),
        "last_actual":     str(df["date"].max().date()),
        "mae":             metrics["mae"],
        "metrics":         metrics,
        "top_features":    top10,
        "dow_baselines":   dow_baselines,
        "top_alert":       top_alert,
        "forecast":        predictions,
    }
    with open(PREDS_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  predictions.json → {PREDS_JSON}  (top_alert={top_alert})")

    # Write model_metrics.json
    full_metrics = {
        "retrain_date":  datetime.date.today().isoformat(),
        "training_rows": len(X_train),
        "holdout_rows":  len(X_test),
        "epoch":         str(epoch.date()),
        "last_actual":   str(df["date"].max().date()),
        **metrics,
        "top_features":  top10,
        "dow_baselines": dow_baselines,
    }
    with open(METRICS, "w") as f:
        json.dump(full_metrics, f, indent=2)

    gha_set("RETRAIN_STATUS", "ok")
    gha_set("RETRAIN_MAE",    metrics["mae"])
    gha_set("RETRAIN_ROWS",   len(X_train))
    print("Done.")


if __name__ == "__main__":
    main()
