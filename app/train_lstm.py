from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from app.preprocess import prepare_data

X, y, scaler = prepare_data("nifty_data.csv")

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=64)
model.save("lstm_model.keras")
joblib.dump(scaler, "scaler.save")
