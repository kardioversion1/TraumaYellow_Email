#!/usr/bin/env python3
"""
traumayellow · retrain.py
Runs every Sunday at 08:00 ET via GitHub Actions.
Joins signals.csv + ed_counts.csv, trains XGBoost on all available labeled data,
evaluates on a 14-day temporal holdout, writes predictions.json + model_metrics.json,
and saves model.pkl.

Sets GitHub Actions environment variables:
  RETRAIN_STATUS   ok | skip | error
  RETRAIN_MAE      float
  RETRAIN_ROWS     int
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

MIN_ROWS = 60
HOLDOUT  = 14


def gha_set(key, value):
    env_file = os.environ.get("GITHUB_ENV")
    if env_file:
        with open(env_file, "a") as f:
            f.write(f"{key}={value}\n")
    print(f"  ENV {key}={value}")


def load_and_join() -> pd.DataFrame:
    ed = pd.read_csv(ED_COUNTS, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if SIGNALS.exists():
        try:
            sig = pd.read_csv(SIGNALS, parse_dates=["date"], on_bad_lines="skip")
        except TypeError:
            sig = pd.read_csv(SIGNALS, parse_dates=["date"], error_bad_lines=False)
        if len(sig) > 0:
            df = ed.merge(sig, on="date", how="left")
            print(f"  Signals joined: {sig['date'].nunique()} signal days, {len(df.columns)} cols total")
            return df
    print("  No signals.csv or empty — training on ed_counts + engineered features only")
    return ed


def build_features(df: pd.DataFrame, epoch: pd.Timestamp):
    df = df.copy()

    # Trend
    df["days_since_epoch"] = (df["date"] - epoch).dt.days
    df["month_index"] = (
        (df["date"].dt.year  - epoch.year)  * 12
      + (df["date"].dt.month - epoch.month)
    )

    # Cyclical
    df["dow_sin"]   = np.sin(2 * np.pi * df["date"].dt.dayofweek / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["date"].dt.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)

    # Calendar
    df["day_of_week"]  = df["date"].dt.dayofweek
    df["is_weekend"]   = (df["date"].dt.dayofweek >= 5).astype(int)
    df["is_monday"]    = (df["date"].dt.dayofweek == 0).astype(int)
    df["is_friday"]    = (df["date"].dt.dayofweek == 4).astype(int)
    df["month"]        = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day

    # Autoregressive lags
    df["visits_lag7"]   = df["total_visits"].shift(7)
    df["visits_lag14"]  = df["total_visits"].shift(14)
    df["visits_roll7"]  = df["total_visits"].shift(1).rolling(7).mean()
    df["visits_roll14"] = df["total_visits"].shift(1).rolling(14).mean()
    df["visits_roll28"] = df["total_visits"].shift(1).rolling(28).mean()

    # Temperature delta (behavioral surge signal)
    if "temp_max_f" in df.columns:
        df["temp_7d_mean"]    = df["temp_max_f"].rolling(7, min_periods=1).mean()
        df["temp_delta"]      = df["temp_max_f"] - df["temp_7d_mean"]
        df["temp_surge_flag"] = (df["temp_delta"] > 15).astype(int)
    else:
        df["temp_delta"] = 0.0
        df["temp_surge_flag"] = 0

    # Ozone WMA (respiratory lag signal — peaks 3-5 days after exposure)
    o3_col = "aqi_o3_ugm3" if "aqi_o3_ugm3" in df.columns else "aqi_o3"
    if o3_col in df.columns:
        o3 = df[o3_col].ffill()
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
        # temperature delta (behavioral surge)
        "temp_delta", "temp_surge_flag",
        # air quality — Open-Meteo AQ (primary, full history)
        "aqi_pm25", "aqi_o3_ugm3",
        # air quality — AirNow EPA (current obs supplement)
        "aqi_o3", "aqi_pm25_airnow", "aqi_overall",
        # ozone WMA respiratory lag
        "aqi_o3_wma",
        # disease surveillance + velocity
        "nssp_flu_pct", "nssp_covid_pct", "nssp_rsv_pct", "nssp_flu_velocity",
        "nwss_percentile",
        # severe weather / infrastructure risk
        "nws_alert_count", "power_risk_flag",
        # community signals
        "crime_violent_7d", "row_permits_catchment",
        # school calendar
        "is_school_out", "is_school_night",
        # events
        "event_attendance",
        # autoregressive
        "visits_lag7", "visits_lag14",
        "visits_roll7", "visits_roll14", "visits_roll28",
    ]

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].copy()
    y = df["total_visits"].copy()
    return X, y, feature_cols, df


def compute_dow_baselines(df: pd.DataFrame) -> dict:
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
    preds     = np.maximum(model.predict(X_test), 0)
    mae       = float(np.mean(np.abs(preds - y_test.values)))
    within_10 = float(np.mean(np.abs(preds - y_test.values) <= 10) * 100)
    surge_mask = y_test.values > (y_test.mean() + y_test.std())
    surge_recall = float(np.mean(preds[surge_mask] > y_test.mean()) * 100) \
        if surge_mask.any() else None
    return {
        "mae":          round(mae, 2),
        "within_10":    round(within_10, 1),
        "surge_recall": round(surge_recall, 1) if surge_recall is not None else None,
    }


# Clinical label mapping for human-readable status briefing
SIGNAL_LABELS = {
    "nssp_flu_velocity":     "Rising Flu Velocity",
    "nssp_flu_pct":          "Elevated Flu Activity",
    "nssp_covid_pct":        "Elevated COVID Activity",
    "nssp_rsv_pct":          "Elevated RSV Activity",
    "power_risk_flag":       "Severe Weather / Power Risk",
    "nws_alert_count":       "Active Weather Alerts",
    "aqi_o3_wma":            "Ozone Lag Signal",
    "aqi_pm25":              "Elevated PM2.5",
    "aqi_o3_ugm3":           "Elevated Ozone",
    "temp_surge_flag":       "Sudden Temperature Shift",
    "temp_delta":            "Temperature Delta",
    "crime_violent_7d":      "Elevated Violent Crime",
    "row_permits_catchment": "Road Disruptions",
    "event_attendance":      "Major Event",
    "is_school_out":         "School Break Period",
    "visits_roll7":          "Recent Volume Trend",
    "month_index":           "Seasonal Trend",
}


def generate_predictions(model, df, epoch, feature_cols, dow_baselines, mae) -> list:
    """Generate predictions always covering today+1 through today+14."""
    last_date  = df["date"].max()
    last_known = df.set_index("date")["total_visits"]

    today  = pd.Timestamp.now().normalize()
    n_days = max(14, (today - last_date).days + 14)

    results = []
    for i in range(1, n_days + 1):
        td = last_date + pd.Timedelta(days=i)
        if td < today:
            continue

        dow              = td.dayofweek
        days_since_epoch = (td - epoch).days
        month_index      = (td.year - epoch.year) * 12 + (td.month - epoch.month)

        visits_lag7   = float(last_known.iloc[-7])  if len(last_known) >= 7  else np.nan
        visits_lag14  = float(last_known.iloc[-14]) if len(last_known) >= 14 else np.nan
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
        last_row = df.iloc[-1]
        for col in feature_cols:
            if col not in base and col in last_row.index:
                base[col] = last_row[col]

        row = pd.DataFrame([base])
        for col in feature_cols:
            if col not in row.columns:
                row[col] = 0
        row = row[feature_cols].fillna(0)

        pred         = float(np.maximum(model.predict(row)[0], 20))
        pred_rounded = round(pred)

        dow_key  = str(dow)
        dow_mean = dow_baselines.get(dow_key, {}).get("mean", pred)
        dow_std  = dow_baselines.get(dow_key, {}).get("std", 10)
        z_score  = round((pred - dow_mean) / dow_std, 2) if dow_std > 0 else 0.0
        alert    = "RED" if z_score >= 2.33 else ("YELLOW" if z_score >= 1.65 else None)

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

    df    = df.dropna(subset=["total_visits"])
    epoch = df["date"].min()

    X, y, feature_cols, df = build_features(df, epoch)

    valid = X["visits_lag14"].notna()
    X, y  = X[valid].fillna(0), y[valid]
    df    = df[valid].reset_index(drop=True)
    print(f"  Training rows after warmup: {len(X)}")

    dow_baselines = compute_dow_baselines(df)
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    print(f"  DOW baselines: " + "  ".join(
        f"{days[i]}={dow_baselines[str(i)]['mean']:.0f}" for i in range(7)
    ))

    split = len(X) - HOLDOUT
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f"  Train: {len(X_train)}  Holdout: {len(X_test)}")

    print("  Training XGBoost...")
    model   = train_model(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    print(f"  MAE={metrics['mae']}  within_10={metrics['within_10']}%  surge_recall={metrics['surge_recall']}%")

    importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
    top10      = dict(sorted(importance.items(), key=lambda x: -x[1])[:10])
    print(f"  Top features: {', '.join(top10.keys())}")

    # Human-readable status briefing from top clinical signals
    briefing_signals = [
        SIGNAL_LABELS[feat] for feat in list(top10.keys())[:3]
        if feat in SIGNAL_LABELS
    ]
    status_briefing = " & ".join(briefing_signals) if briefing_signals else "Baseline Conditions"
    print(f"  Status briefing: {status_briefing}")

    # Save model
    MODEL_PKL.parent.mkdir(exist_ok=True)
    with open(MODEL_PKL, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols, "epoch": str(epoch.date())}, f)
    with open(FEAT_JSON, "w") as f:
        json.dump(feature_cols, f)
    print(f"  Model → {MODEL_PKL}")

    # Generate predictions
    predictions = generate_predictions(model, df, epoch, feature_cols,
                                       dow_baselines, metrics["mae"])
    alerts_3day = [p["alert"] for p in predictions[:3] if p["alert"]]
    top_alert   = "RED" if "RED" in alerts_3day else ("YELLOW" if "YELLOW" in alerts_3day else None)

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
        "status_briefing": status_briefing,
        "forecast":        predictions,
    }

    PREDS_JSON.parent.mkdir(exist_ok=True)
    with open(PREDS_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  predictions.json → {PREDS_JSON}  (top_alert={top_alert})")

    # Model metrics
    with open(METRICS, "w") as f:
        json.dump({
            "retrain_date":  datetime.date.today().isoformat(),
            "training_rows": len(X_train),
            "holdout_rows":  len(X_test),
            "epoch":         str(epoch.date()),
            "last_actual":   str(df["date"].max().date()),
            **metrics,
            "top_features":  top10,
            "dow_baselines": dow_baselines,
        }, f, indent=2)

    gha_set("RETRAIN_STATUS", "ok")
    gha_set("RETRAIN_MAE",    metrics["mae"])
    gha_set("RETRAIN_ROWS",   len(X_train))
    print("Done.")


if __name__ == "__main__":
    main()
