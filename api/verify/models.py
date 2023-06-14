from django.db import models


# Create your models here.
class Document(models.Model):
    file = models.FileField(upload_to='images')
    keyword = models.CharField(max_length=50, null=True, blank=True)
    uploaded = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.pk)