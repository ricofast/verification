"""
URL configuration for coelinks project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
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
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from api.verify.views import FileUpdateView, DocumentScanView, PictureVerifyView, FileUpdatetestView, DocumentVerifiedView
# from rest_framework.routers import DefaultRouter
# from api.verify.views import FileUpdateView, DocumentScanView
# from api.chatbot.views import ChatGenerateView
#
# router = DefaultRouter()
#
# router.register(r"verify/", FileUpdateView, basename="verify")
# router.register(r"chat/", ChatGenerateView, basename="chat")

urlpatterns = [
    path(settings.ADMIN_URL, admin.site.urls),
    path('api/verify/', FileUpdateView.as_view(), name='verify'),
    path('api/testverify/', FileUpdatetestView.as_view(), name='testverify'),
    path('api/scan/', DocumentScanView.as_view(), name='scan'),
    path('api/endverify/', DocumentVerifiedView.as_view(), name='endverify'),
    path('api/checkimage/', PictureVerifyView.as_view(), name='checkimage'),
    path('api/chat', include('api.chatbot.urls', namespace='chat')),
    # path("api/", include(router.urls))
    ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
