import yfinance as yf


def download_data(symbol="^NSEI", interval="1m", period="7d"):
    data = yf.download("NSE:NIFTY50-INDEX", interval="5m", period="60d")
    data = data.dropna()
    data.to_csv("nifty_data.csv")
