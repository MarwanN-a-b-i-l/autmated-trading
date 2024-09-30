import sys
sys.stdout.reconfigure(encoding='utf-8')
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import alpaca_trade_api as tradeapi
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import time

# Download historical stock data for Apple
df = yf.download('AAPL', start='2020-01-01', end='2023-01-01')

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Create a training data set for LSTM
prediction_days = 60
x_train, y_train = [], []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential()

# Add LSTM layers with dropout to avoid overfitting
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Prediction of the next closing price

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

# Your Alpaca API keys
API_KEY = 'PKSUU8MEQBOUK2EOG0CR'
API_SECRET = 'qbLgHybhWy72Ct0bOZVMeREo68suYxHsj09R8ahk'
BASE_URL = 'https://paper-api.alpaca.markets'

# Connect to Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Function to predict next day’s price and make trading decision
def predict_and_trade():
    df = yf.download('AAPL', start='2023-01-01', end='2023-09-01')
    actual_prices = df['Close'].values

    # Prepare the test data
    total_data = pd.concat((df['Close'], df['Close']), axis=0)
    model_inputs = total_data[len(total_data) - len(df) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    # Prepare the input data for the LSTM model
    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Predict stock prices
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Get the latest predicted price
    predicted_price = predicted_prices[-1]

    # Fetch the latest real-time price for comparison
    current_price = actual_prices[-1]

    # Simple buy/sell logic based on predicted price movement
    if predicted_price > current_price:
        print("Predicted upward movement - Buying 1 share of AAPL.")
        api.submit_order(symbol='AAPL', qty=1, side='buy', type='market', time_in_force='gtc')
    else:
        print("Predicted downward movement - Selling 1 share of AAPL.")
        api.submit_order(symbol='AAPL', qty=1, side='sell', type='market', time_in_force='gtc')

# Run prediction and trading logic in a loop
while True:
    try:
        predict_and_trade()  # Run the trading logic
        time.sleep(60 * 60 * 24)  # Run the code once every day
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(60)  # Wait a minute before retrying
