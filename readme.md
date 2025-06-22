# 🧠 Backend - Stock Predictor API (FastAPI + ML)

This is the backend of the Stock Predictor project. It uses FastAPI to serve predictions and other market-related data using trained ML models, the Fyers API, and yFinance.

---

## ⚙️ Setup

### 1. Create and activate a virtual environment (Python 3.11 recommended)

python3.11 -m venv venv  
source venv/bin/activate

### 2. Install dependencies

pip install -r requirements.txt

### 3. Run the FastAPI server

uvicorn app.main:app --reload

---

## 📁 Project Structure

.
├── app  
│   ├── __pycache__/  
│   ├── access_token.txt  
│   ├── auth_config.py            # Stores Fyers app configuration  
│   ├── fyers_client.py           # Handles Fyers REST API logic  
│   ├── fyers_ws.py               # Handles Fyers WebSocket integration  
│   ├── indices.py                # Contains index-related logic  
│   ├── main.py                   # FastAPI app entry point  
│   ├── predictor.py              # Loads and uses ML models for prediction  
│   ├── preprocess.py             # Data preprocessing for predictions  
│   ├── savemodel_scalar.py       # Script to save scalers or models  
│   ├── token_generator.py        # Generates and stores Fyers access token  
│   ├── train_lstm.py             # LSTM training script  
│   └── yf_down_data.py           # Downloads data using yFinance  
├── readme.md                     # This file  
└── requirements.txt              # Python dependencies

---

## 🔐 Environment Setup

Add a `.env` file with:

FYERS_CLIENT_ID=your_client_id  
FYERS_SECRET_KEY=your_secret_key  
FYERS_ACCESS_TOKEN=your_access_token

---

## 📡 API Endpoints (main.py)

- `POST /predict`: Predict stock prices using ML model  
- `GET /stocks/{symbol}`: Fetch historical stock data using yFinance  
- `POST /auth`: Authenticate and get access token from Fyers

---

## 🧠 ML & Tools Used

- TensorFlow (for LSTM model)  
- Scikit-learn (for preprocessing/scaling)  
- yFinance (to fetch stock data)  
- Fyers API v2 (for brokerage data & orders)  
- FastAPI + Uvicorn (for backend API)

---

## 🚀 Run LSTM Training

To retrain the LSTM model, run:

python app/train_lstm.py

---

## 📦 Dependencies (in requirements.txt)

- fastapi  
- uvicorn  
- fyers-apiv2  
- python-dotenv  
- python-multipart  
- jinja2  
- pandas  
- numpy  
- tensorflow  
- scikit-learn  
- joblib  
- yfinance  
- aiofiles

---

## 📜 License

This project is under the MIT License.