from django.db import models

# Create your models here.
class Chat(models.Model):
    user = models.IntegerField()
    question = models.CharField(max_length=255, blank=True, null=True)
    asked = models.DateTimeField(auto_now=True)

    def __str__(self):
        return str(self.user)