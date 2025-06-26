import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import pandas_ta as ta

LOOKBACK = 20
MODEL_PATH = "lstm_model.keras"  # Path to your saved model


def preprocess_for_prediction(symbol="RELIANCE.NS", lookback=60):
    try:
        # Fetch historical stock data
        df = yf.download(symbol, period="5y", interval="1d", auto_adjust=False)
        print(f"[INFO] Downloaded {len(df)} rows from yfinance.")

        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        df.reset_index(inplace=True)

        # Extract required columns
        df = df[["Open", "High", "Low", "Close", "Volume"]]

        # Technical Indicators
        df["EMA_10"] = ta.ema(df["Close"], length=10)
        macd = ta.macd(df["Close"])
        df["MACD"] = macd.iloc[:, 0] if macd is not None and not macd.empty else np.nan

        bb = ta.bbands(df["Close"], length=20)
        if bb is not None and not bb.empty:
            df["BB_upper"] = bb["BBU_20_2.0"]
            df["BB_middle"] = bb["BBM_20_2.0"]
            df["BB_lower"] = bb["BBL_20_2.0"]
        else:
            df["BB_upper"], df["BB_middle"], df["BB_lower"] = np.nan, np.nan, np.nan

        df["SMA_5"] = ta.sma(df["Close"], length=5)
        df["RSI"] = ta.rsi(df["Close"], length=14)
        df["sentiment"] = 0

        print("[DEBUG] NaN counts per column before dropna():")
        print(df.isna().sum())

        df.dropna(inplace=True)
        print("[DEBUG] Length after dropna:", len(df))

        features = ["Open", "High", "Low", "Close",
                    "Volume", "SMA_5", "RSI", "sentiment"]
        print("[DEBUG] Features DataFrame shape:", df[features].shape)

        if df[features].empty or len(df) < lookback:
            raise ValueError(
                "Not enough data to preprocess. Try increasing the time period or reducing LOOKBACK.")

        # Scale and sequence
        data = df[features].values
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        X = []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i - lookback:i])

        return np.array(X), df

    except Exception as e:
        print(f"[ERROR] preprocess_for_prediction() failed: {e}")
        raise


# ------------------------- MAIN -------------------------
if __name__ == "__main__":
    try:
        model = load_model(MODEL_PATH)
        X, df, scaler = preprocess_for_prediction(lookback=LOOKBACK)

        predicted_scaled = model.predict(X)
        predicted_prices = scaler.inverse_transform(
            np.concatenate([predicted_scaled, np.zeros(
                (len(predicted_scaled), 7))], axis=1)
        )[:, 0]

        actual = df["Close"].values[LOOKBACK:]

        plt.figure(figsize=(12, 6))
        plt.plot(actual, label="Actual Price", linewidth=2)
        plt.plot(predicted_prices, label="Predicted Price", linestyle='--')
        plt.title("Price Prediction vs Actual")
        plt.xlabel("Days")
        plt.ylabel("Price (INR)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"[FATAL ERROR] Script failed: {e}")
