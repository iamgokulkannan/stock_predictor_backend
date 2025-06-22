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

<pre>
## 📁 Project Structure

```
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
```
</pre>


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

MIT License

Copyright (c) 2025 Gokul Kannan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
