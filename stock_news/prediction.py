import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# prediction 1
# Fetch live stock market data for a specific stock (e.g., Apple - AAPL)
stock_data = pd.read_csv('static/apple.csv')

# Assuming 'Close' column contains the target variable (stock prices)
X = stock_data[['open', 'high', 'low', 'volume']]  # Features
y = stock_data['close']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

# Make predictions for the next day's stock price
# For simplicity, we'll use the last row of the dataset as input
next_day_features = X_test.iloc[-1].values.reshape(1, -1)
next_day_prediction = rf_regressor.predict(next_day_features)
print("Predicted next day's stock price apple:", next_day_prediction[0])

# Evaluate model performance
y_pred = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print(X)



# prediction 2
# Fetch live stock market data for a specific stock (e.g., Apple - AAPL)
stock_data = pd.read_csv('static/amazone.csv')

# Assuming 'Close' column contains the target variable (stock prices)
X = stock_data[['open', 'high', 'low', 'volume']]  # Features
y = stock_data['close']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

# Make predictions for the next day's stock price
# For simplicity, we'll use the last row of the dataset as input
next_day_features = X_test.iloc[-1].values.reshape(1, -1)
next_day_prediction = rf_regressor.predict(next_day_features)
print("Predicted next day's stock price amazone:", next_day_prediction[0])

# Evaluate model performance
y_pred = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print(X)



#prediction 3
# Fetch live stock market data for a specific stock (e.g., Apple - AAPL)
stock_data = pd.read_csv('static/facebook.csv')

# Assuming 'Close' column contains the target variable (stock prices)
X = stock_data[['open', 'high', 'low', 'volume']]  # Features
y = stock_data['close']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

# Make predictions for the next day's stock price
# For simplicity, we'll use the last row of the dataset as input
next_day_features = X_test.iloc[-1].values.reshape(1, -1)
next_day_prediction = rf_regressor.predict(next_day_features)
print("Predicted next day's stock price facebook:", next_day_prediction[0])

# Evaluate model performance
y_pred = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print(X)



#prediction 4
# Fetch live stock market data for a specific stock (e.g., Apple - AAPL)
stock_data = pd.read_csv('static/google.csv')

# Assuming 'Close' column contains the target variable (stock prices)
X = stock_data[['open', 'high', 'low', 'volume']]  # Features
y = stock_data['close']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

# Make predictions for the next day's stock price
# For simplicity, we'll use the last row of the dataset as input
next_day_features = X_test.iloc[-1].values.reshape(1, -1)
next_day_prediction = rf_regressor.predict(next_day_features)
print("Predicted next day's stock price google:", next_day_prediction[0])

# Evaluate model performance
y_pred = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print(X)



#prediction 5
# Fetch live stock market data for a specific stock (e.g., Apple - AAPL)
stock_data = pd.read_csv('static/ibm.csv')

# Assuming 'Close' column contains the target variable (stock prices)
X = stock_data[['open', 'high', 'low', 'volume']]  # Features
y = stock_data['close']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

# Make predictions for the next day's stock price
# For simplicity, we'll use the last row of the dataset as input
next_day_features = X_test.iloc[-1].values.reshape(1, -1)
next_day_prediction = rf_regressor.predict(next_day_features)
print("Predicted next day's stock price ibm:", next_day_prediction[0])

# Evaluate model performance
y_pred = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print(X)



#prediction 6
# Fetch live stock market data for a specific stock (e.g., Apple - AAPL)
stock_data = pd.read_csv('static/infosys.csv')

# Assuming 'Close' column contains the target variable (stock prices)
X = stock_data[['open', 'high', 'low', 'volume']]  # Features
y = stock_data['close']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

# Make predictions for the next day's stock price
# For simplicity, we'll use the last row of the dataset as input
next_day_features = X_test.iloc[-1].values.reshape(1, -1)
next_day_prediction = rf_regressor.predict(next_day_features)
print("Predicted next day's stock price infosys:", next_day_prediction[0])

# Evaluate model performance
y_pred = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print(X)



#prediction 7
# Fetch live stock market data for a specific stock (e.g., Apple - AAPL)
stock_data = pd.read_csv('static/microsoft.csv')

# Assuming 'Close' column contains the target variable (stock prices)
X = stock_data[['open', 'high', 'low', 'volume']]  # Features
y = stock_data['close']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

# Make predictions for the next day's stock price
# For simplicity, we'll use the last row of the dataset as input
next_day_features = X_test.iloc[-1].values.reshape(1, -1)
next_day_prediction = rf_regressor.predict(next_day_features)
print("Predicted next day's stock price microsoft:", next_day_prediction[0])

# Evaluate model performance
y_pred = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print(X)



#prediction 8
# Fetch live stock market data for a specific stock (e.g., Apple - AAPL)
stock_data = pd.read_csv('static/nvidia.csv')

# Assuming 'Close' column contains the target variable (stock prices)
X = stock_data[['open', 'high', 'low', 'volume']]  # Features
y = stock_data['close']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

# Make predictions for the next day's stock price
# For simplicity, we'll use the last row of the dataset as input
next_day_features = X_test.iloc[-1].values.reshape(1, -1)
next_day_prediction = rf_regressor.predict(next_day_features)
print("Predicted next day's stock price nvidia:", next_day_prediction[0])

# Evaluate model performance
y_pred = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print(X)



#prediction 9
# Fetch live stock market data for a specific stock (e.g., Apple - AAPL)
stock_data = pd.read_csv('static/tcs.csv')

# Assuming 'Close' column contains the target variable (stock prices)
X = stock_data[['open', 'high', 'low', 'volume']]  # Features
y = stock_data['close']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

# Make predictions for the next day's stock price
# For simplicity, we'll use the last row of the dataset as input
next_day_features = X_test.iloc[-1].values.reshape(1, -1)
next_day_prediction = rf_regressor.predict(next_day_features)
print("Predicted next day's stock price tcs:", next_day_prediction[0])

# Evaluate model performance
y_pred = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print(X)



#prediction 10
# Fetch live stock market data for a specific stock (e.g., Apple - AAPL)
stock_data = pd.read_csv('D://Corezone//2024-2025//UC//stock//stock_news//static//tesla.csv')

# Assuming 'Close' column contains the target variable (stock prices)
X = stock_data[['open', 'high', 'low', 'volume']]  # Features
y = stock_data['close']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

# Make predictions for the next day's stock price
# For simplicity, we'll use the last row of the dataset as input
next_day_features = X_test.iloc[-1].values.reshape(1, -1)
next_day_prediction = rf_regressor.predict(next_day_features)
print("Predicted next day's stock price tesla:", next_day_prediction[0])

# Evaluate model performance
y_pred = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print(X)