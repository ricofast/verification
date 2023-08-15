from django.contrib import admin
from .models import Document, AIModel, HeadShot

admin.site.register(Document)
admin.site.register(HeadShot)
admin.site.register(AIModel)