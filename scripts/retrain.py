import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import json
import datetime
import os

# Configuration - adjust these paths to match your repo structure
SIGNALS = 'data/signals.csv'
COUNTS = 'data/ed_counts.csv'
OUTPUT_JSON = 'predictions.json'

def load_and_join():
    """
    Loads signals and hospital data, joining them on date.
    Uses only the first 26 columns to prevent crashes from new data.
    """
    try:
        if not os.path.exists(SIGNALS) or not os.path.exists(COUNTS):
            print("Error: Data files missing.")
            return pd.DataFrame()

        # Safety net: only pull the first 26 columns your model expects
        signals = pd.read_csv(SIGNALS, parse_dates=["date"], usecols=range(26))
        counts = pd.read_csv(COUNTS, parse_dates=["date"])
        
        if not signals.empty and not counts.empty:
            # Merge on date
            df = pd.merge(signals, counts, on="date", how="inner")
            print(f"Successfully joined data. Rows: {len(df)}")
            return df
        else:
            print("Warning: One or more dataframes are empty.")
            return pd.DataFrame()

    except Exception as e:
        print(f"Error during load_and_join: {e}")
        return pd.DataFrame()

def train_model(df):
    """
    Trains an XGBoost model on the joined data.
    """
    if df.empty or len(df) < 10:
        print("Not enough data to train.")
        return None

    # Define features (X) and target (y)
    # Assuming 'date' and the target count 'total_visits' are in your DF
    X = df.drop(columns=['total_visits', 'date'])
    y = df['total_visits']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_test_split=0.2, random_state=42)

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Model trained. Mean Absolute Error: {mae:.2f}")
    
    return model

def generate_predictions(model, df):
    """
    Generates a 7-day forecast and saves to predictions.json.
    """
    if model is None:
        return

    # For this script, we'll use the last available row as a baseline for the forecast
    last_row = df.drop(columns=['total_visits', 'date']).iloc[-1:]
    forecast = []
    today = datetime.date.today()

    for i in range(7):
        pred_date = today + datetime.timedelta(days=i)
        # Simplified: predicting based on the last known state
        pred_val = model.predict(last_row)[0]
        
        forecast.append({
            "date": pred_date.strftime("%Y-%m-%d"),
            "prediction": int(round(pred_val)),
            "level": "Yellow" if pred_val > 100 else "Green" # Adjust logic as needed
        })

    output = {
        "last_updated": datetime.datetime.now().isoformat(),
        "forecast": forecast
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output, f, indent=4)
    print(f"Predictions saved to {OUTPUT_JSON}")

def main():
    print(f"Starting retrain process at {datetime.datetime.now()}")
    
    # 1. Load Data
    df = load_and_join()
    
    # 2. Train Model
    model = train_model(df)
    
    # 3. Generate and Save Predictions
    generate_predictions(model, df)

if __name__ == "__main__":
    main()
