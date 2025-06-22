# app/scheduler.py
import asyncio
from fastapi import BackgroundTasks, APIRouter
from app.predictor import lstm_predict
from datetime import datetime
import json
from app.predictor import lstm_predict
import csv
import os
import time

# Track last predictions in memory
prediction_log = []


async def scheduled_prediction(symbol: str = "NSE:NIFTY50-INDEX"):
    print(
        f"[Scheduler] Running prediction for {symbol} at {datetime.now().strftime('%H:%M:%S')}")
    try:
        result = lstm_predict(symbol)
        prediction_log.append(
            {"timestamp": datetime.now().isoformat(), **result})
        with open("prediction_log.json", "w") as f:
            json.dump(prediction_log, f, indent=2)
    except Exception as e:
        print(f"[Scheduler Error] {e}")


# Async background loop that repeats every N minutes
async def start_scheduler(symbol: str = "NSE:NIFTY50-INDEX", interval_minutes: int = 3):
    while True:
        await scheduled_prediction(symbol)
        await asyncio.sleep(interval_minutes * 60)

router = APIRouter()


@router.get("/auto-predict")
def start_auto_prediction(bg: BackgroundTasks):
    bg.add_task(predict_periodically)
    return {"status": "Scheduler started"}


def predict_periodically():
    from datetime import datetime
    symbols = ["NSE:NIFTY50-INDEX", "NSE:CNXIT-INDEX"]  # Add more if needed
    while True:
        for symbol in symbols:
            result = lstm_predict(symbol)
            log_prediction(result)
        time.sleep(300)  # every 5 mins


def log_prediction(data):
    file = "predictions_log.csv"
    is_new = not os.path.exists(file)
    with open(file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if is_new:
            writer.writeheader()
        writer.writerow(data)
