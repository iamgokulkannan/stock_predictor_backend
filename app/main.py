# app/main.py
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from app.predictor import lstm_predict
import uvicorn
import asyncio
from app.scheduler import start_scheduler

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket for real-time prediction


@app.websocket("/ws/{symbol}")
async def websocket_endpoint(websocket: WebSocket, symbol: str):
    await websocket.accept()
    while True:
        try:
            result = lstm_predict(symbol)
            await websocket.send_json(result)
            await asyncio.sleep(5)  # Fetch every 5 seconds (adjust as needed)
        except Exception as e:
            await websocket.send_json({"error": str(e)})

# Start scheduler on app start


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(start_scheduler(
        symbol="RELIANCE.NS", interval_minutes=3))  # Change symbol as needed

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
