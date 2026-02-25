from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('predict/', views.predict, name='predict'),
    path('history/', views.history_view, name='history'),
    path('delete/<int:pk>/', views.delete_prediction, name='delete_prediction'),
    path('about/', views.about_view, name='about'),
    path('contact/', views.contact_view, name='contact'),
]
