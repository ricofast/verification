from django.urls import re_path
# from django.conf.urls import url
from . import views

app_name = "chat"

urlpatterns = [
    re_path('', views.ChatGenerateView.as_view(), name='chat')
]