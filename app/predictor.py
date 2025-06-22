# app/predictor.py
import os
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import requests
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import SMAIndicator, ADXIndicator
from tensorflow.keras.models import load_model
import openai

# === Load environment variables ===
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY if OPENAI_API_KEY else None

# === Load ML model & scaler ===
model = load_model("lstm_model")
scaler = joblib.load("scaler.save")
analyzer = SentimentIntensityAnalyzer()

with open("confidence.txt") as f:
    confidence = float(f.read().strip())


# === OpenAI Summary ===
def summarize_news(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.7,
            messages=[
                {"role": "system", "content": "Summarize this stock market news in 2 lines"},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content.strip()
    except:
        return None


# === News + Sentiment Fetch ===
def fetch_news_sentiment(query="stock market"):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "pageSize": 5,
        "apiKey": NEWS_API_KEY
    }
    try:
        res = requests.get(url, params=params)
        articles = res.json().get("articles", [])
        scores, headlines, titles = [], [], []
        for article in articles:
            text = article["title"]
            titles.append(text)
            score = analyzer.polarity_scores(text)["compound"]
            scores.append(score)
            headlines.append({"title": text, "url": article["url"]})
        avg_score = round(sum(scores) / len(scores), 3) if scores else 0.0
        label = "Positive" if avg_score > 0.05 else "Negative" if avg_score < -0.05 else "Neutral"
        summary = summarize_news("\n".join(titles)) if OPENAI_API_KEY else None
        return avg_score, label, headlines, summary
    except Exception as e:
        print("[ERROR] News Sentiment:", e)
        return 0.0, "Neutral", [], None


# === Main LSTM Predict Function ===
def lstm_predict(symbol):
    df = yf.download(tickers=symbol, interval="1m", period="90m")
    df = df.dropna()

    # Add indicators
    df["RSI"] = RSIIndicator(df["Close"]).rsi()
    df["SMA"] = SMAIndicator(df["Close"], window=14).sma_indicator()
    df["ADX"] = ADXIndicator(df["High"], df["Low"], df["Close"]).adx()
    df["StochRSI"] = StochRSIIndicator(df["Close"]).stochrsi()
    df.dropna(inplace=True)

    last_60 = df[["Close", "RSI", "SMA", "ADX", "StochRSI"]].values[-60:]
    scaled = scaler.transform(last_60)
    X = np.array([scaled])

    prediction_scaled = model.predict(X)[0][0]
    prediction = scaler.inverse_transform(
        [[prediction_scaled, 0, 0, 0, 0]])[0][0]
    current_price = df["Close"].iloc[-1]

    sentiment_score, sentiment_label, news, summary = fetch_news_sentiment(
        symbol.split(":")[-1])

    # Trade recommendation
    if prediction > current_price and sentiment_label == "Positive":
        recommendation = "BUY"
    elif prediction < current_price and sentiment_label == "Negative":
        recommendation = "SELL"
    else:
        recommendation = "HOLD"

    return {
        "symbol": symbol,
        "predicted_price": round(prediction, 2),
        "current_price": round(current_price, 2),
        "sentiment_score": sentiment_score,
        "sentiment": sentiment_label,
        "recommendation": recommendation,
        "confidence": round(confidence, 2),
        "news": news,
        "summary": summary,
        "timestamp": datetime.datetime.now().isoformat()
    }
