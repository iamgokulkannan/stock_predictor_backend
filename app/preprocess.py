import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def prepare_data(filepath, lookback=60):
    df = pd.read_csv(filepath)
    close_prices = df["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)
    return X, y, scaler
