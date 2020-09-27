from django.db import models
# Create your models here.

class regData(models.Model):
    name = models.CharField(max_length=200)
    reg_date = models.DateTimeField('date registered',auto_now_add=True)
    employee_id = models.SlugField(max_length=200)

    def __str__(self):
        return self.name

    def get_absolute_url(self):
        return reverse('FaceApp:cap_img',
                    args=[self.employee_id])


class detectedData(models.Model):
    # name = models.CharField(max_length=200)
    det_time = models.DateTimeField('time detected')
    employee_id = models.SlugField(max_length=200)

    def __str__(self):
        return self.employee_id