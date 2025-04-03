import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import yfinance as yf
from sklearn.impute import SimpleImputer
from datetime import datetime, timedelta

company = ("AAPL","AMZN","GOOGL","MSFT","NVDA")
for i in company:

    #Fetch stock data for Apple (AAPL) from Yahoo Finance
    symbol = i
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Fetch 1 year of data

    # Download the data
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    # Reset index to make 'Date' a column
    stock_data.reset_index(inplace=True)

    # Handle missing values by imputing with the mean for numerical columns
    imputer = SimpleImputer(strategy='mean')
    numerical_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    stock_data[numerical_columns] = imputer.fit_transform(stock_data[numerical_columns])
    print(stock_data)

    # Feature engineering using pandas
    stock_data['day_of_week'] = stock_data['Date'].dt.dayofweek
    stock_data['day_of_month'] = stock_data['Date'].dt.day
    stock_data['month'] = stock_data['Date'].dt.month
    stock_data['year'] = stock_data['Date'].dt.year

    # Define features and target variable
    X = stock_data[['Open', 'High', 'Low','Volume' , 'day_of_week', 'day_of_month', 'month', 'year']]
    y = stock_data['Close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize the Random Forest Regressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    rf_regressor.fit(X_train, y_train)

    # Make predictions for the next day's stock price
    # For simplicity, we'll use the last row of the dataset as input
    next_day_features = X_test.iloc[-1].values.reshape(1, -1)
    next_day_prediction = rf_regressor.predict(next_day_features)
    print("Predicted next day's stock price for",i,":",next_day_prediction[0])

    # Evaluate model performance
    y_pred = rf_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)



