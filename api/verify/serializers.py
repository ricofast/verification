from rest_framework import serializers
from .models import Document


class FileSerializer(serializers.ModelSerializer):
    class Meta():
        model = Document
        fields = ('user', 'file', 'keyword', 'uploaded')