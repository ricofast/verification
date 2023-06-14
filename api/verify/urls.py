from django.urls import re_path
# from django.conf.urls import url
from . import views

app_name = "verify"

urlpatterns = [
    re_path(r'^upload-scan/$', views.FileUpdateView.as_view(), name='file-upload')
]