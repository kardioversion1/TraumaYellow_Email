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
        print("Not enough data to train. Check signals.csv and ed_counts.csv.")
        return

    # Prepare data for XGBoost
    # Assuming 'total_visits' is your target column
    X = df.drop(columns=['total_visits', 'date'])
    y = df['total_visits']

    # Train model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X, y)

    # Generate a simple 7-day forecast based on the most recent signals
    last_known_signals = X.iloc[-1:]
    forecast = []
    
    for i in range(7):
        pred_date = datetime.date.today() + datetime.timedelta(days=i)
        pred_val = model.predict(last_known_signals)[0]
        
        forecast.append({
            "date": pred_date.strftime("%Y-%m-%d"),
            "prediction": int(round(float(pred_val))),
            "level": "Yellow" if pred_val > 100 else "Green"
        })

    # Save to JSON for the website
    output = {
        "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "forecast": forecast
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output, f, indent=4)
    
    print(f"Success! {OUTPUT_JSON} generated.")

if __name__ == "__main__":
    main()
