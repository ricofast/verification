from django.shortcuts import render
from .serializers import ChatGenerateSerializer
from .models import Chat
from django.views.decorators.csrf import csrf_exempt
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser, FileUploadParser
from rest_framework.response import Response
from rest_framework import status
import re
# Create your views here.


class ChatGenerateView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    # parser_classes = (FileUploadParser,)

    @csrf_exempt
    def post(self, request, *args, **kwargs):
        chat_serializer = ChatGenerateSerializer(data=request.data)
        updated = 'Question added'
        if chat_serializer.is_valid():
            userid = chat_serializer.data['user']
            newquestion = chat_serializer.data['question']
            obj, created = Chat.objects.update_or_create(
                user=userid,
                defaults={'question': newquestion},
            )

            return Response(updated, status=status.HTTP_201_CREATED)
        else:
            return Response(chat_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
