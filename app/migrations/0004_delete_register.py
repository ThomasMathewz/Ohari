# Generated by Django 3.2.25 on 2024-12-21 07:05

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0003_rename_last_name_register_last_name'),
    ]

    operations = [
        migrations.DeleteModel(
            name='Register',
        ),
    ]
