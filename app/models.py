from django.db import models


class Login(models.Model):
    username = models.CharField(max_length=200)
    password = models.CharField(max_length=200)
    user_type = models.CharField(max_length=200)
    
class Register(models.Model):
    LOGIN = models.ForeignKey(Login , on_delete=models.CASCADE)
    first_name = models.CharField(max_length=200)
    last_name = models.CharField(max_length=200)
    email = models.CharField(max_length=200)
    profile = models.CharField(max_length=2000)
    gender = models.CharField(max_length=50)
    phone = models.CharField(max_length=100)
    
class Complaints(models.Model):
    USER = models.ForeignKey(Register, on_delete=models.CASCADE)
    complaint = models.CharField(max_length=20000)
    reply = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    
class Prediction(models.Model):
    USER = models.ForeignKey(Register, on_delete=models.CASCADE)
    stock = models.CharField(max_length=200)
    date = models.CharField(max_length=200)
    amount = models.CharField(max_length=200)