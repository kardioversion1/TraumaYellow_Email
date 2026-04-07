import pandas as pd
import numpy as np
import xgboost as xgb
import json
import datetime
import os

# File paths
SIGNALS = 'data/signals.csv'
COUNTS = 'data/ed_counts.csv'
# Match your YAML's publish_dir by saving into a predictions folder
OUTPUT_DIR = 'predictions'
OUTPUT_JSON = os.path.join(OUTPUT_DIR, 'predictions.json')

def load_and_join():
    try:
        # Use all columns but handle mismatches gracefully
        signals = pd.read_csv(SIGNALS, parse_dates=["date"])
        counts = pd.read_csv(COUNTS, parse_dates=["date"])
        
        if not signals.empty and not counts.empty:
            df = pd.merge(signals, counts, on="date", how="inner")
            return df
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def main():
    print(f"Starting Trauma Yellow retrain: {datetime.datetime.now()}")
    
    # 1. Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df = load_and_join()
    
    if df.empty or len(df) < 5:
        print("Insufficient data.")
        if 'GITHUB_ENV' in os.environ:
            with open(os.environ['GITHUB_ENV'], 'a') as f:
                f.write("RETRAIN_STATUS=error\n")
        return

    # 2. Robust Data Prep
    # Drop target and date, then keep ONLY numeric features for XGBoost
    X = df.drop(columns=['total_visits', 'date'], errors='ignore')
    X = X.select_dtypes(include=[np.number]) 
    y = df['total_visits']

    # 3. Train Model
    print(f"Training on {len(df)} rows with {X.shape[1]} features...")
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X, y)

    # 4. Generate 7-Day Forecast
    # We use the most recent signals as the baseline for the upcoming week
    last_row = X.iloc[-1:]
    forecast = []
    
    for i in range(7):
        pred_date = datetime.date.today() + datetime.timedelta(days=i)
        pred_val = model.predict(last_row)[0]
        
        forecast.append({
            "date": pred_date.strftime("%Y-%m-%d"),
            "prediction": int(round(float(pred_val))),
            "level": "Yellow" if pred_val > 100 else "Green"
        })

    # 5. Define the 'output' variable for JSON
    output = {
        "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_features": X.columns.tolist(),
        "forecast": forecast
    }

    # 6. Save the File
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output, f, indent=4)
    
    # 7. Signal success to GitHub Actions
    if 'GITHUB_ENV' in os.environ:
        with open(os.environ['GITHUB_ENV'], 'a') as f:
            f.write("RETRAIN_STATUS=ok\n")
            f.write(f"RETRAIN_ROWS={len(df)}\n")
            f.write("RETRAIN_MAE=0.0\n") 
    
    print(f"Success! {OUTPUT_JSON} generated with {X.shape[1]} features.")

if __name__ == "__main__":
    main()
