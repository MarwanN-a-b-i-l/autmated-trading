
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# Download AAPL stock data from Yahoo Finance
aapl_data = yf.download('AAPL', start='2022-01-01', end='2024-09-26')

# Plot the closing price of AAPL
plt.figure(figsize=(14, 7))
plt.plot(aapl_data['Close'], label='AAPL Closing Price', color='blue')
plt.title('Apple (AAPL) Stock Closing Price (2022 - 2024)')
plt.xlabel('Date')
plt.ylabel('Price in USD')
plt.legend(loc='upper left')

# Optional: Plot a Moving Average to Identify Trends
aapl_data['50_MA'] = aapl_data['Close'].rolling(window=50).mean()  # 50-day moving average
aapl_data['200_MA'] = aapl_data['Close'].rolling(window=200).mean()  # 200-day moving average

plt.plot(aapl_data['50_MA'], label='50-Day Moving Average', color='orange')
plt.plot(aapl_data['200_MA'], label='200-Day Moving Average', color='green')

plt.legend(loc='upper left')
plt.grid(True)
plt.show()
