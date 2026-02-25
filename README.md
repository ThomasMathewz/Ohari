#  Ohari — AI Powered Stock Analysis & Prediction System

---

Ohari is a **Django-based Stock Market Web Application** that combines:

*  Real-time stock data
*  Stock news integration
*  LSTM-based deep learning price prediction
*  AI-powered stock assistant
*  Admin & User management system
*  Complaint & feedback system

This project demonstrates full-stack development with **Machine Learning, Sentiment Analysis, and AI integration**.

---

#  Key Features

##  Authentication & User Roles

* User Registration with profile upload
* Login & Logout system
* Role-based access:

  * **Admin**
  * **User**
  * **Blocked User**
* Change Password
* Forgot Password with OTP (Email-based)
* Profile Editing

---

##  Admin Functionalities

* View all registered users
* Block / Unblock users
* View and reply to complaints
* View stock-related news
* View real-time stock prices (AAPL, GOOGL, MSFT)
* Change admin password

---

##  User Functionalities

* View real-time stock prices
* View stock market news (filter by company)
* View historical stock data
* Predict next-day stock price using LSTM
* Send complaints to admin
* View admin replies
* AI Stock Assistant (Ask stock-related questions)
* Edit profile
* Change password

---

#  Stock Data Integration

Stock prices are fetched using:

* `yfinance` (Yahoo Finance API)

Tracked stocks (default):

* AAPL
* GOOGL
* MSFT

---

#  News Integration

News is fetched using:

* NewsAPI
* AlphaVantage (for sentiment-based news)

Users can filter news by companies:

* Tesla
* Apple
* Microsoft
* Google
* Amazon
* Nvidia
* Facebook
* IBM
* Infosys

---

#  LSTM Stock Price Prediction

The prediction pipeline includes:

### 1️ Historical Data Fetching

* 6 months stock data using `yfinance`

### 2️ Sentiment Analysis

* News sentiment extracted using `TextBlob`
* Polarity score added as feature

### 3️ Data Preparation

* MinMax Scaling
* Lookback window = 60 days
* Features:

  * Close Price
  * Sentiment Score

### 4️ Model Architecture

```
LSTM (50 units, return_sequences=True)
Dropout (0.2)
LSTM (50 units)
Dropout (0.2)
Dense (25)
Dense (1)
```

Optimizer: Adam
Loss: Mean Squared Error
Epochs: 20

### 5️ Output

* Predicts next day closing price
* Generates matplotlib graph
* Graph converted to Base64 and embedded in HTML

---

#  AI Stock Assistant

Integrated with:

* OpenRouter API
* DeepSeek Model (`deepseek-r1-distill-qwen-32b`)

Users can:

* Ask stock-related queries
* Get AI-generated financial insights

Response is returned as formatted HTML via AJAX.

---

#  Project Modules Overview

## Authentication

* Login
* Register
* Forgot Password (OTP via Email)
* Session Management

## Admin Panel

* User Management
* Complaint Management
* News Viewing

## User Dashboard

* Stock Prices
* Historical Data
* Prediction Engine
* AI Assistant
* Profile Management

---

#  Technology Stack

## Backend

* Django
* Python

## Machine Learning

* TensorFlow / Keras
* LSTM Networks
* Scikit-learn (MinMaxScaler)
* TextBlob (Sentiment Analysis)

## Data APIs

* Yahoo Finance (yfinance)
* NewsAPI
* AlphaVantage

## Frontend

* HTML
* CSS
* Django Templates
* JavaScript (AJAX for AI assistant)

## Visualization

* Matplotlib

## Database

* Django ORM (SQLite by default)

---

#  Installation Guide

## 1️ Clone the Repository

```bash
git clone https://github.com/ThomasMathewz/Ohari.git
cd Ohari
```

## 2️ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

## 3️ Install Requirements

```bash
pip install -r requirements.txt
```

---

#  Setup Environment Variables

You must configure:

* NewsAPI Key
* AlphaVantage API Key
* OpenRouter API Key
* Email settings (for OTP system)

Update in:

* settings.py
* Views where API keys are used

> Important: Never expose API keys in production.

---

#  Run the Project

```bash
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```

Visit:

```
http://127.0.0.1:8000/
```

---

#  Prediction Flow Summary

User selects stock →
Fetch historical data →
Fetch sentiment →
Prepare dataset →
Train LSTM →
Predict next-day price →
Display result + Graph

---

#  Security Notes

* Passwords are stored as plain text (Needs improvement)
* API keys are hardcoded (Should be moved to environment variables)
* Model retrains on every prediction (Can be optimized)

This project is primarily for academic and demonstration purposes.

---

#  Project Domain

* Artificial Intelligence
* Deep Learning
* FinTech
* Web Development
* Time Series Forecasting
* NLP (Sentiment Analysis)

---

#  Future Improvements

* Use Django authentication system with password hashing
* Store trained LSTM model instead of retraining each request
* Add stock portfolio management
* Improve UI/UX
* Deploy on cloud (AWS / Render / Railway)
* Add REST APIs

---

#  Author

Thomas Mathew

B.Tech Graduate

Aspiring AI & Software Developer

---


