# Required Libraries
import pandas as pd
import pandas_ta as ta
import numpy as np
import yfinance as yf
from textblob import TextBlob
import requests
from sklearn.preprocessing import MinMaxScaler
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ------------------------------------------
# 🚀 1. Fetch Historical Stock Data (6 Months)
# ------------------------------------------
def get_stock_data(ticker, period='6mo', interval='1d'):
    data = yf.download(ticker, period=period, interval=interval)
    data['Date'] = data.index
    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    data.reset_index(drop=True, inplace=True)
    return data

# ------------------------------------------
# 📰 2. Fetch and Perform Sentiment Analysis on News
# ------------------------------------------
def get_news_sentiment(ticker):
    api_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey=YOUR_API_KEY'
    response = requests.get(api_url)
    
    if response.status_code == 200:
        news_data = response.json()
        articles = news_data.get('feed', [])
        sentiments = []
        
        for article in articles:
            summary = article.get('summary', '')
            sentiment_score = get_sentiment(summary)
            sentiments.append(sentiment_score)
        
        # Calculate average sentiment for the last few days
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        return avg_sentiment
    else:
        print("Failed to fetch news data.")
        return 0

# ------------------------------------------
# 🧠 3. Sentiment Analysis Using TextBlob
# ------------------------------------------
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# ------------------------------------------
# 📊 Calculate Technical Indicators
# ------------------------------------------
def add_technical_indicators(data):
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')  # Convert to numeric
    data['Close'] = data['Close'].bfill()
    data['Close'] = data['Close'].ffill()
    # Forward fill if necessary

    # Add Technical Indicators
    data['SMA_50'] = ta.sma(data['Close'], length=50).fillna(0)
    data['SMA_200'] = ta.sma(data['Close'], length=200)
    data['EMA_12'] = ta.ema(data['Close'], length=12).fillna(0)
    data['EMA_26'] = ta.ema(data['Close'], length=26).fillna(0)
    data['RSI'] = ta.rsi(data['Close'], length=14).fillna(0)
    macd = ta.macd(data['Close'], fast=12, slow=26, signal=9).fillna(0)
    data['MACD'] = macd['MACD_12_26_9'].fillna(0)
    data['MACD_signal'] = macd['MACDs_12_26_9'].fillna(0)
    bollinger = ta.bbands(stock_data['Close'], length=20)
    data['UpperBand'] = bollinger['BBU_20_2.0'].fillna(0)
    data['MiddleBand'] = bollinger['BBM_20_2.0'].fillna(0)
    data['LowerBand'] = bollinger['BBL_20_2.0'].fillna(0)
    
    return data

# ------------------------------------------
# 🧩 4. Prepare Data for LSTM Model
# ------------------------------------------
def prepare_data(data, sentiment_score, lookback=60):
    # Add Sentiment as a Feature
    data['Sentiment'] = sentiment_score

    # Use only 'Close' price for prediction + sentiment
    data['Close'] = data['Close'].astype(float)
    features = ['Close', 'Sentiment', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_signal', 'UpperBand', 'MiddleBand', 'LowerBand']
    dataset = data[features].values
    
    # Scale the Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    # Create Training Data
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, :])  # Last 'lookback' days of data
        y.append(scaled_data[i, 0])  # Predict 'Close' price

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], len(features)))  # Reshape for LSTM
    return X, y, scaler

# ------------------------------------------
# 📈 Build LSTM Model with Extra Features
# ------------------------------------------
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    
    model.add(Dense(25))
    model.add(Dense(1))  # Predict Close price
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# ------------------------------------------
# 🧪 6. Train the Model
# ------------------------------------------
def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
    return model

# ------------------------------------------
# 🔮 Predict Next Day Price with Indicators
# ------------------------------------------
def predict_next_day(model, data, scaler, lookback=60):
    last_days_data = data[-lookback:][['Close', 'Sentiment', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_signal', 'UpperBand', 'MiddleBand', 'LowerBand']].values
    scaled_data = scaler.transform(last_days_data)
    
    X_test = np.array([scaled_data])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
    
    predicted_price_scaled = model.predict(X_test)
    predicted_price = scaler.inverse_transform(np.array([[predicted_price_scaled[0][0], 0, 0, 0, 0, 0, 0]]))[:, 0]
    return predicted_price[0]

# ------------------------------------------
# 🚀 8. Run the Full Pipeline
# ------------------------------------------
ticker='AAPL'
# Get Stock Data and Sentiment Score

stock_data = get_stock_data(ticker)
stock_data.columns = stock_data.columns.get_level_values(0)
print(stock_data.head())
sentiment_score = get_news_sentiment(ticker)
technical_indicators= add_technical_indicators(stock_data)

# Prepare Data
X, y, scaler = prepare_data(stock_data, sentiment_score, technical_indicators)

# Build and Train LSTM Model with New Input Shape
model = build_lstm_model((X.shape[1], X.shape[2]))
model = train_model(model, X, y, epochs=50, batch_size=32)

# Predict Next Day Price
predicted_price = predict_next_day(model, stock_data, scaler)
print(f"📊 Predicted next day price for {ticker}: ${predicted_price:.2f}")

# Plot Results
plt.figure(figsize=(14, 6))
plt.plot(stock_data['Date'][-100:], stock_data['Close'][-100:], label='Actual Price')
plt.plot(stock_data['Date'][-1:], [predicted_price], 'ro', label='Predicted Price')
plt.title(f"{ticker} - Actual vs Predicted Price")
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

