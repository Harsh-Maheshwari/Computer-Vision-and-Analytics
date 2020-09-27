# Generated by Django 3.0.6 on 2020-06-29 19:39

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='detectedData',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('det_time', models.DateTimeField(verbose_name='time detected')),
                ('employee_id', models.SlugField(max_length=200)),
            ],
        ),
        migrations.CreateModel(
            name='regData',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=200)),
                ('reg_date', models.DateTimeField(auto_now_add=True, verbose_name='date registered')),
                ('employee_id', models.SlugField(max_length=200)),
            ],
        ),
    ]
