# === enhanced_lstm_trainer.py ===
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib
import requests
from dotenv import load_dotenv
import openai
import os
from scipy.ndimage import gaussian_filter1d
import yfinance as yf

load_dotenv()

LOOKBACK = 60
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY if OPENAI_API_KEY else None


def download_data_yahoo(symbol="^NSEI", period="90d", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval)
    df.reset_index(inplace=True)
    df.to_csv("nifty_data.csv", index=False)
    return "nifty_data.csv"


def fetch_sentiment(symbol):
    company_name = symbol.split(".")[0]  # From RELIANCE.NS -> RELIANCE
    url = f"https://newsapi.org/v2/everything?q={company_name}&apiKey={NEWS_API_KEY}&pageSize=5"
    try:
        res = requests.get(url)
        articles = res.json().get("articles", [])
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(
            a["title"])["compound"] for a in articles]
        summary = summarize_news(
            "\n".join(a["title"] for a in articles)) if OPENAI_API_KEY else None
        return np.mean(scores) if scores else 0.0, summary
    except Exception as e:
        print("[ERROR] Sentiment fetch:", e)
        return 0.0, None


def summarize_news(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.7,
            messages=[
                {"role": "system",
                    "content": "Summarize this stock market news in 2-3 lines"},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("[ERROR] OpenAI Summary:", e)
        return None


def load_and_preprocess(filepath, symbol="RELIANCE.NS"):
    # Read CSV and skip bad lines if any (prevents malformed rows)
    df = pd.read_csv(filepath, on_bad_lines='skip')

    # Ensure numeric types — handle bad data if any
    cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows with NaN values
    df = df.dropna()

    # Add technical indicators
    df["EMA_10"] = ta.ema(df["Close"], length=10)
    df["MACD"] = ta.macd(df["Close"]).iloc[:, 0]
    bb = ta.bbands(df["Close"], length=20)
    df["BB_upper"], df["BB_middle"], df["BB_lower"] = bb["BBU_20_2.0"], bb["BBM_20_2.0"], bb["BBL_20_2.0"]
    df["SMA_5"] = ta.sma(df["Close"], length=5)
    df["RSI"] = ta.rsi(df["Close"], length=14)

    # Drop any resulting NaN from indicator calculations
    df = df.dropna()

    # Add sentiment score
    sentiment_score, _ = fetch_sentiment(symbol)
    df["sentiment"] = sentiment_score

    # Prepare features
    features = ["Open", "High", "Low", "Close",
                "Volume", "SMA_5", "RSI", "sentiment"]
    data = df[features].values

    # Normalize features
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    # Build sequences
    X, y = [], []
    for i in range(LOOKBACK, len(scaled)):
        X.append(scaled[i - LOOKBACK:i])
        y.append(scaled[i, 3])  # Target: Close price

    X, y = np.array(X), np.array(y)
    return X, y, scaler, df


def build_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(
        LSTM(64, return_sequences=True), input_shape=input_shape))
    model.add(GRU(64, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def evaluate_accuracy(y_true, y_pred):
    direction_match = np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))
    accuracy = np.mean(direction_match) * 100
    return round(accuracy, 2)


# === Main Execution ===
symbol = "RELIANCE.NS"  # changeable
data_file = download_data_yahoo(symbol)
X, y, scaler, df = load_and_preprocess(data_file, symbol)

model = build_model((X.shape[1], X.shape[2]))
callbacks = [
    EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-6)
]

history = model.fit(X, y, epochs=20, batch_size=64, verbose=1,
                    validation_split=0.1, callbacks=callbacks)
loss = model.evaluate(X, y, verbose=0)
confidence = max(0, 1 - loss)

model.save("lstm_model.keras")
joblib.dump(scaler, "scaler.save")
with open("confidence.txt", "w") as f:
    f.write(str(confidence))

preds = model.predict(X).flatten()
smoothed_preds = gaussian_filter1d(preds, sigma=2)
accuracy = evaluate_accuracy(y, smoothed_preds)

print(f"[INFO] Model confidence: {round(confidence * 100, 2)}%")
print(f"[INFO] Prediction direction accuracy: {accuracy}%")

pd.DataFrame({
    "timestamp": df.index[-len(y):],
    "actual": y,
    "predicted": smoothed_preds
}).to_csv("logs.csv", index=False)

print("[SUCCESS] Model, confidence, and logs saved.")
