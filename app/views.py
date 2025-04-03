from django.shortcuts import render,HttpResponse
from .models import *
import datetime
from django.core.files.storage import FileSystemStorage
from app.models import Login 
# Create your views here.
def index(request):
    return render(request,"public/index.html")

def login(request):
    if 'submit' in request.POST:
        username=request.POST['username']
        password=request.POST['password']

        if Login.objects.filter(username=username,password=password).exists():
            obj=Login.objects.get(username=username,password=password)
            request.session['login_id']=obj.pk
            print(request.session['login_id'],'==============================================')
            
            if obj.user_type=='admin':
                request.session['log']="in"
                return HttpResponse(f"<script>alert('Admin login successfully');window.location='/admin_home'</script>")
            
            elif obj.user_type=='user':
                request.session['log']="in"
            
            elif obj.user_type=='blocked':
                request.session['log']="out"
                return HttpResponse(f"<script>alert('You have been blocked !');window.location='/login'</script>")
            
            q=Register.objects.get(LOGIN_id=request.session["login_id"])
            if q:
                request.session['user_id']=q.pk
                return HttpResponse(f"<script>alert('User login successfully');window.location='/user_home'</script>")
            else:
                return HttpResponse(f"<script>alert('invalid username or password');window.location='/login'</script>")
            
        else:
            return HttpResponse(f"<script>alert('invalid username or password');window.location='/login'</script>")
        
    return render(request,'public/login.html')

def register(request):
    if 'submit' in request.POST:
        first_name = request.POST['fname']
        last_name = request.POST['lname']
        username = request.POST['username']
        email = request.POST['email']
        profile = request.FILES['profile']
        gender = request.POST['gender']
        phone_number = request.POST['phone']
        password = request.POST['password']
        import datetime
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+".jpg"
        fs = FileSystemStorage()
        fp=fs.save(date,profile)
        obj=Login(username=username,password=password,user_type='user')
        obj.save()
        obj1=Register(first_name=first_name ,last_name=last_name,email=email,profile=fs.url(fp),gender=gender,phone=phone_number , LOGIN_id=obj.pk)
        obj1.save()
        return HttpResponse(f"<script>alert('User Registered Successfully');window.location='/login'</script>")
    return render(request,"public/register.html")
def admin_home(request):
    first_name = "Admin"
    import requests
    from django.shortcuts import render

    # Function to fetch stock prices (Replace with a different API if needed)
    import yfinance as yf

    def get_stock_prices():
        stock_symbols = ["AAPL", "GOOGL", "MSFT"]  # Add the stocks you want to track
        stock_data = {}
        
        for symbol in stock_symbols:
            try:
                # Download the latest stock data for the symbol
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d")  # Get the latest day's price data
                
                # Check if data is available
                if not data.empty:
                    stock_data[symbol] = round(data['Close'].iloc[-1], 2)  # Get the last closing price
                else:
                    stock_data[symbol] = "N/A"
            
            except Exception as e:
                stock_data[symbol] = f"Error: {str(e)}"
        
        return stock_data


    # Function to fetch stock-related news
    def get_stock_news():
        API_KEY = "d1bc02bacbbc4d50a7f66a15d5c38cb0"
        NEWS_URL = f"https://newsapi.org/v2/everything?q=stock market&apiKey={API_KEY}"
        
        response = requests.get(NEWS_URL)
        news_data = response.json()
        articles = news_data.get("articles", [])[:5]  # Fetch top 5 news articles
        
        return articles
    stock_prices = get_stock_prices()
    news_articles = get_stock_news()
    
    return render(request,"admin/admin_home.html", {
        'name':first_name,
        "stock_prices": stock_prices,
        "articles": news_articles,
    })
    
def user_home(request):
    name = Register.objects.get(LOGIN_id=request.session['login_id'])
    first_name = name.first_name
    import requests
    from django.shortcuts import render

    # Function to fetch stock prices (Replace with a different API if needed)
    import yfinance as yf

    def get_stock_prices():
        stock_symbols = ["AAPL", "GOOGL", "MSFT"]  # Add the stocks you want to track
        stock_data = {}
        
        for symbol in stock_symbols:
            try:
                # Download the latest stock data for the symbol
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d")  # Get the latest day's price data
                
                # Check if data is available
                if not data.empty:
                    stock_data[symbol] = round(data['Close'].iloc[-1], 2)  # Get the last closing price
                else:
                    stock_data[symbol] = "N/A"
            
            except Exception as e:
                stock_data[symbol] = f"Error: {str(e)}"
        
        return stock_data

    # Function to fetch stock-related news
    def get_stock_news():
        API_KEY = "d1bc02bacbbc4d50a7f66a15d5c38cb0"
        NEWS_URL = f"https://newsapi.org/v2/everything?q=stock market&apiKey={API_KEY}"
        
        response = requests.get(NEWS_URL)
        news_data = response.json()
        articles = news_data.get("articles", [])[:5]  # Fetch top 5 news articles
        
        return articles
    stock_prices = get_stock_prices()
    news_articles = get_stock_news()
    
    return render(request, "user/user_home.html", {
        'name':first_name,
        "stock_prices": stock_prices,
        "articles": news_articles,
    })
def logout(request):
    request.session['log']="out"
    request.session.flush()
    return HttpResponse(f"<script>alert('Logged out successfully');window.location='/'</script>")

def admin_view_users(request):
    data =Register.objects.all()
    return render(request,"admin/admin_view_users.html",{'data':data})

def block(request , id):
    Login.objects.filter(id=id).update(user_type='blocked')
    return HttpResponse(f"<script>alert('Blocked successfully');window.location='/admin_view_users'</script>")

def unblock(request , id):
    Login.objects.filter(id=id).update(user_type='user')
    return HttpResponse(f"<script>alert('Unblocked successfully');window.location='/admin_view_users'</script>")

def admin_view_news(request):
    import requests
    from django.shortcuts import render

    API_KEY = "d1bc02bacbbc4d50a7f66a15d5c38cb0"
    company = request.GET.get("company", "stock market")  # Default to "stock market" if no company is selected
    NEWS_URL = f"https://newsapi.org/v2/everything?q={company}&apiKey={API_KEY}"

    response = requests.get(NEWS_URL)
    news_data = response.json()

    articles = news_data.get("articles", [])[:100]  # Get up to 100 articles

    # List of companies for drop-down
    companies = ["Tesla", "Apple", "Microsoft", "Google", "Amazon", "Nvidia","Facebook","IBM","Infosys"]

    return render(request, "user/user_view_news.html", {
        "articles": articles,
        "companies": companies,
        "selected_company": company
    })


def admin_view_complaints(request):
    a =Complaints.objects.all()
    return render(request,"admin/admin_view_complaints.html",{'a':a})

def admin_send_reply(request,id):
    data = Complaints.objects.get(id=id)
    if 'send' in request.POST:
        reply = request.POST['reply']
        Complaints.objects.filter(id=id).update(reply=reply)
        return HttpResponse(f"<script>alert('Reply sent successfully');window.location='/admin_view_complaints'</script>")
    return render(request,"admin/admin_send_reply.html",{'data':data})

def user_edit_profile(request):
    uid=request.session['user_id'] 
    data = Register.objects.get(id=uid)
    if 'submit' in request.POST:
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        email = request.POST['email']
        gender = request.POST['gender']
        phone_number = request.POST['phone']
        if 'profile' in request.FILES:
            profile = request.FILES['profile']
            date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+".jpg"
            fs = FileSystemStorage()
            fp=fs.save(date,profile)
            data.profile=fs.url(fp)
        data.first_name=first_name
        data.last_name=last_name
        data.email=email
        data.gender=gender
        data.phone=phone_number
        data.save()
        return HttpResponse(f"<script>alert('Profile Updated');window.location='/user_view_profile'</script>")
    return render(request,"user/user_edit_profile.html",{'data':data})

def user_view_profile(request):
    uid=request.session['user_id'] 
    data = Register.objects.get(id=uid)
    return render(request,"user/user_view_profile.html",{'data':data})

def user_view_news(request):
    import requests
    from django.shortcuts import render

    API_KEY = "d1bc02bacbbc4d50a7f66a15d5c38cb0"
    company = request.GET.get("company", "stock market")  # Default to "stock market" if no company is selected
    NEWS_URL = f"https://newsapi.org/v2/everything?q={company}&apiKey={API_KEY}"

    response = requests.get(NEWS_URL)
    news_data = response.json()

    articles = news_data.get("articles", [])[:100]  # Get up to 100 articles

    # List of companies for drop-down
    companies = ["Tesla", "Apple", "Microsoft", "Google", "Amazon", "Nvidia","Facebook","IBM","Infosys"]

    return render(request, "user/user_view_news.html", {
        "articles": articles,
        "companies": companies,
        "selected_company": company
    })


def user_view_stock(request):
    data =Register.objects.all()
    return render(request,"user/user_view_stock.html",{'data':data})

import datetime
def user_send_complaints(request):
    uid=request.session['user_id'] 
    if 'submit' in request.POST:
        complaint = request.POST['complaint']
        date = datetime.datetime.today()
        a=Complaints(complaint=complaint,reply='pending',date=date,USER_id=uid)
        a.save()
        return HttpResponse(f"<script>alert('Complaint sent successfully');window.location='/user_send_complaints'</script>")
    return render(request,"user/user_send_complaints.html",{'id':id})

def user_view_historical_stock(request):
    data =Register.objects.all()
    return render(request,"user/user_view_historical_stock.html",{'data':data})

def predict_u(request):
    data =request.POST['stock']
    
    # Required Libraries
    import pandas as pd
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
    import base64
    from io import BytesIO

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
    # 🧩 4. Prepare Data for LSTM Model
    # ------------------------------------------
    def prepare_data(data, sentiment_score, lookback=60):
        # Add Sentiment as a Feature
        data['Sentiment'] = sentiment_score
        
        # Use only 'Close' price for prediction + sentiment
        data['Close'] = data['Close'].astype(float)
        dataset = data[['Close', 'Sentiment']].values
        
        # Scale the Data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        
        # Create Training Data
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, :])  # Last 'lookback' days of data
            y.append(scaled_data[i, 0])  # Predict 'Close' price

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 2))  # Reshape for LSTM
        return X, y, scaler

    # ------------------------------------------
    # 📈 5. Build LSTM Model
    # ------------------------------------------
    def build_lstm_model(input_shape):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        
        model.add(Dense(25))
        model.add(Dense(1))  # Predict 1 value (Close price)
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model

    # ------------------------------------------
    # 🧪 6. Train the Model
    # ------------------------------------------
    def train_model(model, X_train, y_train, epochs=10, batch_size=32):
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
        return model

    # ------------------------------------------
    # 🔮 7. Predict Next Day Price
    # ------------------------------------------
    def predict_next_day(model, data, scaler, lookback=60):
        last_days_data = data[-lookback:][['Close', 'Sentiment']].values
        scaled_data = scaler.transform(last_days_data)
        
        X_test = np.array([scaled_data])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 2))
        
        predicted_price_scaled = model.predict(X_test)
        predicted_price = scaler.inverse_transform(np.array([[predicted_price_scaled[0][0], 0]]))[:, 0]
        return predicted_price[0]

    # ------------------------------------------
    # 8.Convert Matplotlib plot to Base64 to embed in HTML
    # ------------------------------------------
    def plot_to_base64(stock_data, predicted_price, ticker):
        plt.figure(figsize=(14, 6))
        plt.plot(stock_data['Date'][-100:], stock_data['Close'][-100:], label='Actual Price')
        plt.plot(stock_data['Date'][-1:], [predicted_price], 'ro', label='Predicted Price')
        plt.title(f"{ticker} - Actual vs Predicted Price")
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()

        # Save plot to BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Encode plot to base64 string
        graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        # Return data URI to embed in HTML
        return f"data:image/png;base64,{graph_base64}"
    # ------------------------------------------
    # 🚀 9. Run the Full Pipeline
    # ------------------------------------------
    ticker=data
    # Get Stock Data and Sentiment Score
    stock_data = get_stock_data(ticker)
    sentiment_score = get_news_sentiment(ticker)

    # Prepare Data
    X, y, scaler = prepare_data(stock_data, sentiment_score)

    # Build and Train LSTM Model
    model = build_lstm_model((X.shape[1], X.shape[2]))
    model = train_model(model, X, y, epochs=20, batch_size=32)

    # Predict Next Day Price
    predicted_price = predict_next_day(model, stock_data, scaler)
    print(f"📊 Predicted next day price for {ticker}: ${predicted_price:.2f}")
    # Generate graph in base64
    graph_url = plot_to_base64(stock_data, predicted_price, ticker)
     
    return render(request,"user/user_view_historical_stock.html",{'data':data,"result":predicted_price,'graph_url': graph_url})

def user_view_reply(request):
    uid=request.session['user_id']
    reply = Complaints.objects.filter(USER_id=uid)
    return render(request,"user/user_view_reply.html",{'reply':reply})
# --------------------------------------------------------------------------------------------------------------------------
from openai import OpenAI
from django.shortcuts import render
from django.http import JsonResponse

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",  # Replace with your actual API key
)

def stock_ai_assistant(request):
    if request.method == "POST":
        user_query = request.POST.get("query", "").strip()

        if not user_query:
            return JsonResponse({"error": "Please enter a stock-related question."}, status=400)

        try:
            completion = client.chat.completions.create(
                model="deepseek/deepseek-r1-distill-qwen-32b:free",  # Use the correct model
                messages=[{"role": "user", "content": user_query}],
                timeout=10  # Prevent long waiting times
            )

            ai_response = completion.choices[0].message.content

            # Convert markdown-like text to proper HTML
            formatted_response = (
                ai_response.replace("###", "<h3>").replace("", "<b>").replace("---", "<hr>")
                .replace("* ", "<li>").replace("\n", "<br>")  # Convert lists and line breaks
            )

            return JsonResponse({"advice": formatted_response})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return render(request,'user/stock_ai_assistant.html')

def user_change_password(request):
    if request.method == 'POST':
        old_password = request.POST.get('old_password')
        new_password = request.POST.get('new_password')
        confirm_password = request.POST.get('confirm_password')
        uid = request.session['login_id']
        user = Login.objects.get(id=uid)
        if old_password == user.password:
            if new_password != old_password:
                if new_password == confirm_password:
                    user.password = new_password
                    user.save()
                    return HttpResponse("<script>alert('Password changed successfully');window.location='/login'</script>")
                else:
                    return HttpResponse("<script>alert('New passwords do not match');window.location='/user_change_password'</script>")
            else:
                return HttpResponse("<script>alert('New password cannot be the same as the old password');window.location='/user_change_password'</script>")
        else:
            return HttpResponse("<script>alert('Old password is incorrect');window.location='/user_change_password'</script>")
    
    return render(request, 'user/user_change_password.html')

def admin_change_password(request):
    if request.method == 'POST':
        old_password = request.POST.get('old_password')
        new_password = request.POST.get('new_password')
        confirm_password = request.POST.get('confirm_password')
        user = Login.objects.get(user_type='admin')
        if old_password == user.password:
            if new_password != old_password:
                if new_password == confirm_password:
                    user.password = new_password
                    user.save()
                    return HttpResponse("<script>alert('Password changed successfully');window.location='/login'</script>")
                else:
                    return HttpResponse("<script>alert('New passwords do not match');window.location='/admin_change_password'</script>")
            else:
                return HttpResponse("<script>alert('New password cannot be the same as the old password');window.location='/admin_change_password'</script>")
        else:
            return HttpResponse("<script>alert('Old password is incorrect');window.location='/admin_change_password'</script>")
    
    return render(request, 'admin/admin_change_password.html')

from django.core.mail import send_mail
from django.conf import settings
import random

def forgot_password(request):
    # data =Register.objects.all()
    # return render(request,"public/user_forgot_password.html",{'data',data})
    if 'submit' in request.POST:
        otp=0
        uname=request.POST['username']
        email=request.POST['email']
        request.session['username']=uname
        request.session['email']=email
        otp = random.randint(1000, 9999)
        request.session['otp']=otp
        subject = 'Your OTP for Registration'
        message = f"""
        Thank you for registering!
        
        Please use the following OTP to verify your account:
        - OTP: {otp}
        """
        from_email = settings.EMAIL_HOST_USER
        recipient_list = [email]

    try:
        send_mail(subject, message, from_email, recipient_list)
    except Exception as e:
        return HttpResponse(f"<script>alert('Error sending email: {e}');window.location='/login';</script>")

    return render(request,'public/otp_for_forgot_password.html')

def otp_for_forgot_password(request):
    return render (request,'public/forgot_password.html')



    
