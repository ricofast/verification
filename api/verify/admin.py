from django.contrib import admin
from .models import Document, AIModel, HeadShot, AIModelLoaded, KerasModelLoaded

admin.site.register(Document)
admin.site.register(HeadShot)
admin.site.register(AIModel)
admin.site.register(AIModelLoaded)
admin.site.register(KerasModelLoaded)