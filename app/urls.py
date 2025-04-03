"""stock_news URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path , include
from .import views

urlpatterns = [
    path('',views.index),
    path('login/', views.login),
    path('register/', views.register,),
    path('admin_home/', views.admin_home),
    path('user_home/', views.user_home),
    path('logout/', views.logout),
    path('admin_view_news/',views.admin_view_news),
    path('admin_view_users/',views.admin_view_users),
    path('admin_view_complaints/',views.admin_view_complaints),
    path('admin_change_password/',views.admin_change_password),
    path('admin_send_reply/<id>/',views.admin_send_reply),
    path('user_view_profile/',views.user_view_profile),
    path('user_edit_profile/',views.user_edit_profile),
    path('user_send_complaints/',views.user_send_complaints),
    path('user_view_stock/',views.user_view_stock),
    path('user_view_news/',views.user_view_news),
    path('user_view_historical_stock/',views.user_view_historical_stock),
    path('predict_u/',views.predict_u),
    path('stock_ai_assistant/',views.stock_ai_assistant, name='stock_ai_assistant'),
    path('user_view_reply/',views.user_view_reply),
    path('user_change_password/',views.user_change_password),
    path('forgot_password/',views.forgot_password),
    path('otp_for_forgot_password',views.otp_for_forgot_password),
    path('block/<id>/',views.block),
    path('unblock/<id>/',views.unblock),
]