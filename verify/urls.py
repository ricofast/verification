from django.urls import path, include, re_path
# from django.conf.urls import url
from . import views

app_name = "verify"

urlpatterns = [
    re_path(r'^upload/$', views.FileView.as_view(), name='file-upload')
]