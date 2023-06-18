from rest_framework import serializers
from .models import Chat


class ChatGenerateSerializer(serializers.ModelSerializer):
    class Meta():
        model = Chat
        fields = ('user', 'question', 'asked')