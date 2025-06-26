# app/scheduler.py
import asyncio
from app.predictor import lstm_predict
from datetime import datetime
import json
import csv
import os
import time

# Store predictions in memory
prediction_log = []


async def scheduled_prediction(symbol: str = "RELIANCE.NS"):
    print(
        f"[Scheduler] Predicting for {symbol} at {datetime.now().strftime('%H:%M:%S')}")
    try:
        result = lstm_predict(symbol)
        prediction_log.append(
            {"timestamp": datetime.now().isoformat(), **result})

        # Save JSON log
        with open("prediction_log.json", "w") as f:
            json.dump(prediction_log, f, indent=2)

        # Optional: also save to CSV
        log_prediction(result)

    except Exception as e:
        print(f"[Scheduler Error] {e}")


async def start_scheduler(symbol: str = "RELIANCE.NS", interval_minutes: int = 3):
    while True:
        await scheduled_prediction(symbol)
        await asyncio.sleep(interval_minutes * 60)

# For manual HTTP trigger (not required if using startup scheduler)


def predict_periodically():
    symbols = ["RELIANCE.NS", "TCS.NS"]  # Add more as needed
    while True:
        for symbol in symbols:
            result = lstm_predict(symbol)
            log_prediction(result)
        time.sleep(300)  # Every 5 minutes


def log_prediction(data):
    file = "predictions_log.csv"
    is_new = not os.path.exists(file)
    with open(file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if is_new:
            writer.writeheader()
        writer.writerow(data)
