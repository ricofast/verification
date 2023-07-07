from django.db import models

def document_directory_path(instance, filename):

    return 'images/user_{0}/{1}'.format(instance.id, filename)

# Create your models here.
class Document(models.Model):
    user = models.IntegerField()
    file = models.FileField(upload_to='images/')
    keyword = models.CharField(max_length=50, null=True, blank=True)
    uploaded = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.pk)