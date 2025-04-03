# Not official Code

import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# Load the dataset
stock_data = pd.read_csv(r'C:\Users\thoma\OneDrive\Desktop\stock_news\stock_news\static')

# Calculate technical indicators using pandas_ta
stock_data['SMA_50'] = ta.sma(stock_data['close'], length=50)  # Simple Moving Average
stock_data['SMA_200'] = ta.sma(stock_data['close'], length=200)
stock_data['EMA_12'] = ta.ema(stock_data['close'], length=12)
stock_data['EMA_26'] = ta.ema(stock_data['close'], length=26)
stock_data['RSI'] = ta.rsi(stock_data['close'], length=14)
macd = ta.macd(stock_data['close'], fast=12, slow=26, signal=9)  # MACD indicators
stock_data['MACD'] = macd['MACD_12_26_9']
stock_data['MACD_signal'] = macd['MACDs_12_26_9']
bollinger = ta.bbands(stock_data['close'], length=20)
stock_data['UpperBand'] = bollinger['BBU_20_2.0']
stock_data['MiddleBand'] = bollinger['BBM_20_2.0']
stock_data['LowerBand'] = bollinger['BBL_20_2.0']

# Drop NaN values introduced by indicators
stock_data = stock_data.dropna()

# Features and Target
X = stock_data[['open', 'high', 'low', 'volume', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_signal', 'UpperBand', 'MiddleBand', 'LowerBand']]
y = stock_data['close']

# Scale the features and target
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Reshape X for LSTM
time_steps = 1
X_scaled = X_scaled.reshape(X_scaled.shape[0], time_steps, X_scaled.shape[1])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=64, activation='tanh', return_sequences=True, input_shape=(time_steps, X.shape[1])))  # First LSTM
model.add(Dropout(0.2))  # Dropout to prevent overfitting
model.add(LSTM(units=32, activation='tanh'))  # Second LSTM
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Output layer

# Optimizer with ExponentialDecay learning rate schedule
lr_schedule = ExponentialDecay(initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.9)
adam_optimizer = Adam(learning_rate=lr_schedule)

# Compile the model with Huber loss
model.compile(optimizer=adam_optimizer, loss='huber_loss')

# Callbacks: EarlyStopping and ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_lstm_model.h5', monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=100, batch_size=32, verbose=1,
                    callbacks=[early_stopping, model_checkpoint])

# Predict the next day's stock price
try:
    next_day_features = X_test[-1].reshape(1, time_steps, X.shape[1])  # Ensure proper shape
    next_day_prediction = model.predict(next_day_features)
    next_day_prediction = scaler_y.inverse_transform(next_day_prediction)  # Rescale to original range

    # Display the prediction
    print("Predicted next day's stock price:", next_day_prediction[0][0])
except Exception as e:
    print(f"Error during prediction: {e}")

# Evaluate model performance
y_pred = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
