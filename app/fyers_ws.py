from fyers_api.Websocket import ws
from token_generator import load_token
import json

data_store = {"ltp": {}}

access_token = load_token()


def custom_message(msg):
    symbol = msg['symbol']
    ltp = msg.get('ltp', None)
    if ltp:
        data_store["ltp"][symbol] = ltp
        print(f"[LIVE] {symbol}: {ltp}")


def onerror(msg):
    print("[ERROR] WebSocket:", msg)


def onclose(msg):
    print("[CLOSED] WebSocket:", msg)


def start_socket():
    fyers_socket = ws.FyersSocket(
        access_token=access_token,     # ✅ Required field
        run_background=True,
        log_path=""
    )

    fyers_socket.websocket_data = custom_message
    fyers_socket.onerror = onerror
    fyers_socket.onclose = onclose

    symbols = ['NSE:NIFTY50-INDEX', 'NSE:NIFTYBANK-INDEX']
    fyers_socket.subscribe(symbols=symbols, data_type="SymbolData")

    fyers_socket.keep_running()
    return fyers_socket
