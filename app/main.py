from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from app.fyers_ws import data_store, start_socket
from app.predictor import lstm_predict
import uvicorn
import asyncio
from app.scheduler import start_scheduler

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
            result = lstm_predict(symbol)
            await websocket.send_json(result)
        except Exception as e:
            await websocket.send_json({"error": str(e)})


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(start_scheduler(
        symbol="NSE:NIFTY50-INDEX", interval_minutes=3))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
