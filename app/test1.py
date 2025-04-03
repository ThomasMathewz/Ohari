
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout,  Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2, l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
from textblob import TextBlob
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ----------------------------------------------------------------------------------------------------------------------------------------------

# Get the current timestamp in the required format
stock_data = [[]]
df = [[]]
ticker= 'AMZN'
date = datetime.now().strftime("%Y%m%dT0000")
print(date)
# Construct the API URL  (NP16WASG3Q4KMP82) (WM25MK3BXKH6YHTQ) (J41PMO618QEF01VL)
key='WM25MK3BXKH6YHTQ'

# ------------------------------------------
# 🧠 3. Sentiment Analysis Using TextBlob
# ------------------------------------------
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# 📰 2. Fetch and Perform Sentiment Analysis on News (1 Year)
# ------------------------------------------
def get_news_sentiment(ticker):
    days=365
    api_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={key}'
    response = requests.get(api_url)
    
    if response.status_code == 200:
        news_data = response.json()
        articles = news_data.get('feed', [])
        daily_sentiment = {}

        for article in articles:
            summary = article.get('summary', '')
            sentiment_score = get_sentiment(summary)
            
            # Extract and format the article's published date
            time_published = article.get('time_published', '')
            if time_published:
                try:
                    # Correct format parsing
                    date_obj = datetime.strptime(time_published, "%Y%m%dT%H%M%S")
                    date = date_obj.strftime('%Y-%m-%d')
                except ValueError as e:
                    print(f"Error parsing date: {e}")
                    continue  # Extract YYYY-MM-DD format
                if date in daily_sentiment:
                    daily_sentiment[date].append(sentiment_score)
                else:
                    daily_sentiment[date] = [sentiment_score]

        # Calculate average sentiment per day
        daily_avg_sentiment = []
        for date, sentiments in daily_sentiment.items():
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            daily_avg_sentiment.append({'Date': date, 'Sentiment': avg_sentiment})
        
        # ✅ Check if data exists before creating the DataFrame
        if daily_avg_sentiment:
            sentiment_df = pd.DataFrame(daily_avg_sentiment)
            sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
            sentiment_df = sentiment_df.sort_values(by='Date', ascending=False)

            # Return only the last 'days' days
            sentiment_df = sentiment_df.head(days).reset_index(drop=True)
            return sentiment_df
        else:
            print("⚠️ No sentiment data available for the given ticker.")
            return pd.DataFrame(columns=['Date', 'Sentiment'])

    else:
        print("❌ Failed to fetch news data.")
        return {}

# -----------------------------------------------------------------------------------------------------------------------------------
# Fetch stock data using yfinance
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data[['Close', 'Open', 'High', 'Low', 'Volume',]]

# Get daily sentiment scores 
df = get_news_sentiment(ticker)

if not df.empty:
    print(df.head())
else:
    print("No data available.")

#     print(f"{date}: {score:.4f}")
end_date = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
stock_data = fetch_stock_data(ticker, start_date, end_date)

# Print the DataFrame to check if it contains data
print("Data after loading:")
# print(stock_data.to_string())

# Check if the DataFrame is empty
if stock_data.empty:
    print("The DataFrame is empty. Please check your data loading process.")
else:
    # Fill NaN values using backfill/forward fill
    stock_data['Close'] = stock_data['Close'].bfill().ffill()
    # Check for NaN values in the 'Close' column
    if stock_data['Close'].isna().any().item():
        print("NaN values found in 'Close' column")
    else:
        print("No NaN values in 'Close' column")

    # Ensure there are enough data points
    print(f"Number of data points: {len(stock_data)}")

    # Calculate technical indicators
    stock_data['SMA_50'] = ta.sma(stock_data['Close'], length=50)
    stock_data['SMA_200'] = ta.sma(stock_data['Close'], length=200)
    stock_data['EMA_12'] = ta.ema(stock_data['Close'], length=12)
    stock_data['EMA_26'] = ta.ema(stock_data['Close'], length=26)
    stock_data['RSI'] = ta.rsi(stock_data['Close'], length=14)

    # Calculate MACD 
    macd = ta.macd(stock_data['Close'], fast=12, slow=26, signal=9)

    # Check if macd is not None before assignment
    if macd is not None and not macd.empty:
        stock_data['MACD'] = macd['MACD_12_26_9']
        stock_data['MACD_signal'] = macd['MACDs_12_26_9']
        stock_data['MACD_hist'] = macd['MACDh_12_26_9']
    
    # Calculate Bollinger Bands
    bollinger = ta.bbands(stock_data['Close'], length=20, std=2)

    # Check if bollinger is valid
    if bollinger is not None and not bollinger.empty:
        stock_data['UpperBand'] = bollinger['BBU_20_2.0'].fillna(0)
        stock_data['MiddleBand'] = bollinger['BBM_20_2.0'].fillna(0)
        stock_data['LowerBand'] = bollinger['BBL_20_2.0'].fillna(0)

    # Convert dictionary to DataFrame
    sentiment_df = pd.DataFrame(list(df.items()), columns=['Date', 'Sentiment'])

    # Reset index to remove multi-level index if any
    stock_data = stock_data.reset_index()
    sentiment_df = sentiment_df.reset_index()

    # Merge on 'Date' after resetting index
    stock_data = stock_data.merge(sentiment_df, on='Date', how='left')

    # Set 'Date' back as index if needed
    stock_data.set_index('Date', inplace=True)


# Add lag features
stock_data['Close_Lag1'] = stock_data['Close'].shift(1)
stock_data['Close_Lag2'] = stock_data['Close'].shift(2)
stock_data['Daily_Change'] = (stock_data['Close'] - stock_data['Open']) / stock_data['Open']
print(stock_data.head())
# Drop NaN values
# stock_data = stock_data.dropna()

# Features and Target 
X = stock_data['Open', 'High', 'Low', 'Volume','Sentiment','Close_Lag1', 'Close_Lag2', 'Daily_Change','SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'RSI', 
                'MACD', 'MACD_signal','MACD_hist', 'UpperBand', 'MiddleBand', 'LowerBand','Close_Lag1', 'Close_Lag2', 'Daily_Change',]
y = stock_data['Close']
print(X.head())

# Scale the data
scaler_X = StandardScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Create sequences
def create_sequences(X, y, time_steps=10):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

time_steps = 2
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False)

# Define the LSTM model
model = Sequential([
    # First Bidirectional LSTM layer
    Bidirectional(LSTM(units=128, return_sequences=True, input_shape=(time_steps, X.shape[1]))),  
    Dropout(0.3),  

    # Second Bidirectional LSTM layer
    Bidirectional(LSTM(units=64, return_sequences=True)),  
    Dropout(0.3),

    # Third Bidirectional LSTM layer
    Bidirectional(LSTM(units=32, return_sequences=True)),  
    Dropout(0.3),

    # Fourth LSTM layer (not bidirectional)
    LSTM(units=16),  
    Dropout(0.3),

    # Fully connected output layer
    Dense(units=1)  
])          

# Learning rate scheduler
lr_schedule = ExponentialDecay(initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.95)
optimizer = Adam(learning_rate=lr_schedule)

# Compile the model
model.compile(optimizer=optimizer, loss='huber_loss')
# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_lstm_model.h5', monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                    epochs=100, batch_size=64, verbose=1, 
                    callbacks=[early_stopping, model_checkpoint])

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_inverse = scaler_y.inverse_transform(y_pred)
y_test_inverse = scaler_y.inverse_transform(y_test)

try:
    next_day_features = X_test[-1].reshape(1, time_steps, X.shape[1])  # Ensure proper shape
    next_day_prediction = model.predict(next_day_features)
    next_day_prediction = scaler_y.inverse_transform(next_day_prediction)  # Rescale to original range

    # Display the prediction
    print("Predicted next day's stock price:", next_day_prediction[0][0])
except Exception as e:
    print(f"Error during prediction: {e}")

mse = mean_squared_error(y_test_inverse, y_pred_inverse)
mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot predictions vs actual values
plt.figure(figsize=(12, 6))
plt.plot(y_test_inverse, label="Actual Prices")
plt.plot(y_pred_inverse, label="Predicted Prices")
plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Time Steps")
plt.ylabel("Stock Price")
plt.legend()
plt.show()
