from rest_framework import serializers
from .models import Document, HeadShot


class FileSerializer(serializers.ModelSerializer):
    class Meta():
        model = Document
        fields = ('user', 'file', 'keyword', 'uploaded')


class FileScanSerializer(serializers.ModelSerializer):
    class Meta():
        model = Document
        fields = ('user', 'keyword')


class HeadShotSerializer(serializers.ModelSerializer):
    class Meta():
        model = HeadShot
        fields = ('user', 'file', 'uploaded')