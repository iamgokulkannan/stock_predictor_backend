from fastapi import FastAPI, BackgroundTasks
from apscheduler.schedulers.background import BackgroundScheduler
from app.predictor import lstm_predict
from datetime import datetime
import csv
import os

app = FastAPI()
scheduler = BackgroundScheduler()
scheduler.start()

log_file = "logs.csv"


def auto_predict():
    result = lstm_predict("NSE:NIFTY50-INDEX")
    print("[Auto Predict]", result)
    if not os.path.exists(log_file):
        with open(log_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(result.keys())
    with open(log_file, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result.values())


scheduler.add_job(auto_predict, "interval", minutes=5)


@app.get("/")
def root():
    return {"status": "Scheduler Running"}
