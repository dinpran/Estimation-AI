from django.contrib import admin
from django.urls import path
from data import views

urlpatterns = [
    path('', views.index, name="index"),
    path('emissions/', views.emissions, name='emissions'),  # Corrected URL mapping
    path('cancer/', views.cancer, name='cancer'),  # Corrected URL mapping
    path('estimation/', views.estimation, name='estimation'), 
    path('churn/', views.churn, name='churn'),
    path('consumption/', views.consumption, name='consumption'),
    path('about/', views.about, name='about'),
    path('contact_us/', views.contact_us, name='contact_us'),
    path('price/', views.price, name='price'),
    path('predict/', views.predict_inputs, name='predict_inputs'),
    path('predict_cancer/', views.predict_cancer, name='predict_cancer'),  # Corrected URL mapping
    path('predict_estimation/', views.predict_inputs, name='predict_estimation'),
    path('predict_house_price/', views.predict_house_price, name='predict_house_price'),
    path('predict_churn/', views.predict_churn, name='predict_churn'),
    path('predict_consumption/', views.predict_consumption, name='predict_consumption'),
    path('yacht/', views.yacht, name='yacht'),
    path('predict_yacht/', views.predict_yacht_hydrodynamics, name='predict_yacht'),  
]
