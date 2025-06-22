# app/fyers_client.py
from fyers_api import fyersModel

client_id = "YOUR_CLIENT_ID"
access_token = "YOUR_ACCESS_TOKEN"

fyers = fyersModel.FyersModel(
    client_id=client_id,
    token=access_token,
    log_path=""
)


def get_stock_price(symbol="NSE:JKTYRE-EQ"):
    res = fyers.quotes({"symbols": symbol})
    try:
        return res["d"][0]["v"]["ltp"]  # Last traded price
    except Exception as e:
        print("[ERROR]", e)
        return 0.0
