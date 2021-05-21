from django.db import models

# Create your models here.
class Image(models.Model):
    image = models.ImageField(upload_to="img/")
    identify_category = (('text',"TEXT"), ('object',"OBJECT"))
    identify = models.CharField(max_length=10,choices=identify_category,null=True)
    