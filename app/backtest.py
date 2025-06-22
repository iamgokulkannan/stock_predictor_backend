# === backtest.py ===
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import SMAIndicator, ADXIndicator

scaler = joblib.load("scaler.save")
model = load_model("lstm_model")
LOOKBACK = 60


def add_indicators(df):
    df["RSI"] = RSIIndicator(df["Close"]).rsi()
    df["SMA"] = SMAIndicator(df["Close"], window=14).sma_indicator()
    df["ADX"] = ADXIndicator(df["High"], df["Low"], df["Close"]).adx()
    df["StochRSI"] = StochRSIIndicator(df["Close"]).stochrsi()
    return df.dropna()


def backtest(file="nifty_data.csv"):
    df = pd.read_csv(file)
    df = add_indicators(df)
    X, y = [], []
    for i in range(LOOKBACK, len(df) - 1):
        features = df[["Close", "RSI", "SMA", "ADX",
                       "StochRSI"]].iloc[i-LOOKBACK:i].values
        X.append(scaler.transform(features))
        y.append(df["Close"].iloc[i+1] > df["Close"].iloc[i])  # True = BUY

    X, y = np.array(X), np.array(y)
    preds = model.predict(X)
    pred_labels = preds[:, 0] > scaler.transform(df[["Close"]])[LOOKBACK:, 0]
    accuracy = accuracy_score(y, pred_labels)
    print(f"Backtest Accuracy: {accuracy:.2%}")

    df_result = pd.DataFrame({
        "Actual": y,
        "Predicted": pred_labels
    })
    df_result.to_csv("backtest_results.csv", index=False)
    print("[✓] Saved backtest_results.csv")


def simulate_trades(df, threshold=0.5):
    df = df.dropna()
    df['Signal'] = 0
    df.loc[(df['Predicted'] > df['Actual']) & (
        df['Sentiment'] == 'Positive'), 'Signal'] = 1
    df.loc[(df['Predicted'] < df['Actual']) & (
        df['Sentiment'] == 'Negative'), 'Signal'] = -1

    df['Correct'] = (df['Signal'] > 0) & (
        df['Actual'].shift(-1) > df['Actual'])
    df['Correct'] |= (df['Signal'] < 0) & (
        df['Actual'].shift(-1) < df['Actual'])
    accuracy = df['Correct'].mean()
    return df, accuracy


if __name__ == "__main__":
    backtest()
