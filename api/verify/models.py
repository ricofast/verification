from django.db import models

def document_directory_path(instance, filename):

    return 'images/user_{0}/{1}'.format(instance.user, filename)

# Create your models here.
class Document(models.Model):
    user = models.IntegerField()
    file = models.FileField(upload_to=document_directory_path)
    keyword = models.CharField(max_length=50, null=True, blank=True)
    uploaded = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.user)


class AIModel(models.Model):
    name = models.CharField(max_length=50, null=True, blank=True)
    file = models.FileField(upload_to="aimodels")
    description = models.CharField(max_length=250, null=True, blank=True)
    uploaded = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.name)
