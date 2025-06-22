import requests
from app.auth_config import AUTH_CONFIG
from textblob import TextBlob


def fetch_news_sentiment(symbol):
    try:
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={AUTH_CONFIG['NEWS_API_KEY']}"
        res = requests.get(url)
        articles = res.json().get("articles", [])[:5]
        combined = " ".join(article['title'] for article in articles)
        sentiment = TextBlob(combined).sentiment.polarity
        label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
        return sentiment, label, articles
    except Exception as e:
        return 0.0, "Neutral", []
