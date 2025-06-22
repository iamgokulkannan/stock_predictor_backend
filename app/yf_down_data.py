import yfinance as yf


def download_data(symbol="^NSEI", interval="1m", period="7d"):
    data = yf.download(tickers=symbol, interval=interval, period=period)
    data = data.dropna()
    data.to_csv("nifty_data.csv")
