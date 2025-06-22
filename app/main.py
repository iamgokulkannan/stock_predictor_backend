from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from app.fyers_ws import data_store, start_socket
from app.predictor import lstm_predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

socket = start_socket()  # Run on startup


@app.websocket("/ws/{symbol}")
async def websocket_endpoint(websocket: WebSocket, symbol: str):
    await websocket.accept()
    while True:
        try:
            pred = lstm_predict(symbol)
            await websocket.send_json({"symbol": symbol, "prediction": pred})
        except Exception as e:
            await websocket.send_json({"error": str(e)})
