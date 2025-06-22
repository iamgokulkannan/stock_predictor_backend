import numpy as np
import yfinance as yf
import joblib
from app.preprocess import prepare_data
from tensorflow.keras.models import load_model

model = load_model("lstm_model")
scaler = joblib.load("scaler.save")


def lstm_predict(symbol):
    df = yf.download(tickers=symbol, interval="1m", period="90m")
    df = df.dropna()
    last_60 = df["Close"].values[-60:].reshape(-1, 1)

    scaled = scaler.transform(last_60)
    X = np.array([scaled])
    prediction_scaled = model.predict(X)[0][0]
    prediction = scaler.inverse_transform([[prediction_scaled]])[0][0]

    model.save("lstm_model")
    joblib.dump(scaler, "scaler.save")

    return round(prediction, 2)
