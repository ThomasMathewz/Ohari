# Not Official code

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.layers import  LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. Fetch Stock Data
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data[['Close', 'Open', 'High', 'Low', 'Volume']]  # Add additional features

# 2. Prepare Data for LSTM (Predicting Next Day's Close)
def prepare_data(data, look_back):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(look_back, len(scaled_data) - 1):  # Adjusted to exclude the last day's value
        X.append(scaled_data[i-look_back:i, :])  # Use multiple features (Open, High, Low, Close, Volume)
        y.append(scaled_data[i+1, 0])  # Predict the next day's closing price

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))  # Reshape for multiple features
    return X, y, scaler

# 3. Build the LSTM Model with Enhanced Architecture
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))  # Reduced units to prevent overfitting
    model.add(Dropout(0.2))  # Reduced dropout for better generalization
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Output the next day's closing price
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')  # Optimizer with adjusted learning rate
    return model

# 4. Main Function
def main():
    # Parameters
    ticker = 'TSLA'  # Change to your preferred stock ticker
    start_date = '2015-01-01'
    end_date = '2023-12-31'
    look_back = 360  # Number of previous days to consider
    model_path = 'static/lstm_model.keras'

    # Fetch and Prepare Data
    data = fetch_stock_data(ticker, start_date, end_date)
    X, y, scaler = prepare_data(data.values, look_back)

    # Split Data into Training and Testing Sets
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Check if model exists, otherwise build a new one
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = load_model(model_path)
    else:
        print("Creating a new model...")
        # Build and Train the Model
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

    # Early stopping and learning rate scheduler
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

    history = model.fit(
        X_train, y_train,
        epochs=100, batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr]
    )
    
    # Save the model after training
    model.save(model_path)
    print("Model saved to", model_path)

    # Predict and Plot Results
    predictions = model.predict(X_test)

    # Inverse transform the predictions (only the close price)
    predictions_rescaled = scaler.inverse_transform(
        np.concatenate((predictions, np.zeros((predictions.shape[0], 4))), axis=1)
    )[:, 0]

    true_prices = scaler.inverse_transform(
        np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 4))), axis=1)
    )[:, 0]

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(true_prices, label='Actual Next Day Prices')
    plt.plot(predictions_rescaled, label='Predicted Next Day Prices')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title(f'{ticker} Stock Price Prediction (Next Day Close)')
    plt.show()

    # Plot training vs validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.show()

if __name__ == "__main__":
    main()
