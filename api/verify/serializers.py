from rest_framework import serializers
from .models import Document, HeadShot


class FileSerializer(serializers.ModelSerializer):
    class Meta():
        model = Document
        fields = ('user', 'verified', 'file', 'uploaded')


class FileScanSerializer(serializers.ModelSerializer):
    class Meta():
        model = Document
        fields = ('user', 'keyword', 'name', 'name_checked', 'dob', 'dob_checked',
                  'grade', 'grade_checked', 'scanned', 'verified')


class HeadShotSerializer(serializers.ModelSerializer):
    file = serializers.FileField(max_length=None, use_url=True, allow_null=True, required=False)
    class Meta():
        model = HeadShot
        fields = ('user', 'file', 'uploaded')
