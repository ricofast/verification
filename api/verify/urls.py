from django.urls import re_path
# from django.conf.urls import url
from . import views

app_name = "verify"

urlpatterns = [
    re_path('', views.FileUpdateView.as_view(), name='file-upload'),
    re_path('scan/', views.DocumentScanView.as_view(), name='file-scan'),
]