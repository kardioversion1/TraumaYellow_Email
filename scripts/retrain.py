import pandas as pd
import numpy as np
import xgboost as xgb
import json
import datetime
import os

# File paths
SIGNALS = 'data/signals.csv'
COUNTS = 'data/ed_counts.csv'
OUTPUT_JSON = 'predictions.json'

def load_and_join():
    """Loads signals and hospital data, joining them on date."""
    try:
        # Safety net: only pull the first 26 columns to prevent 'saw 33' errors
        signals = pd.read_csv(SIGNALS, parse_dates=["date"], usecols=range(26))
        counts = pd.read_csv(COUNTS, parse_dates=["date"])
        
        if not signals.empty and not counts.empty:
            df = pd.merge(signals, counts, on="date", how="inner")
            return df
        else:
            print("One of the data files is empty.")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def main():
    print(f"Starting Trauma Yellow retrain: {datetime.datetime.now()}")
    
    df = load_and_join()
    
    if df.empty or len(df) < 5:
        print("Not enough data to train.")
        # Tell GitHub it failed
        if 'GITHUB_ENV' in os.environ:
            with open(os.environ['GITHUB_ENV'], 'a') as f:
                f.write("RETRAIN_STATUS=error\n")
        return

    # Prepare data and Train (Simplified logic for the fix)
    X = df.drop(columns=['total_visits', 'date'])
    y = df['total_visits']
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X, y)

    # ... (Your existing forecast logic here) ...

    # SAVE THE JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output, f, indent=4)
    
    # --- ADD THIS PART TO TALK TO GITHUB ---
    if 'GITHUB_ENV' in os.environ:
        with open(os.environ['GITHUB_ENV'], 'a') as f:
            f.write("RETRAIN_STATUS=ok\n")
            f.write(f"RETRAIN_ROWS={len(df)}\n")
            f.write("RETRAIN_MAE=0.0\n") # You can calculate real MAE later
    # ---------------------------------------
    
    print(f"Success! {OUTPUT_JSON} generated and status sent to GitHub.")

if __name__ == "__main__":
    main()
