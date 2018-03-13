# -*- coding: utf-8 -*-
# Generated by Django 1.11.10 on 2018-03-08 10:28
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('main', '0023_auto_20180308_1028'),
    ]

    operations = [
        migrations.CreateModel(
            name='Bagging',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=30)),
                ('time', models.CharField(max_length=30)),
                ('index', models.CharField(max_length=30)),
            ],
        ),
        migrations.CreateModel(
            name='Boosting',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=30)),
                ('time', models.CharField(max_length=30)),
                ('index', models.CharField(max_length=30)),
            ],
        ),
        migrations.CreateModel(
            name='Data',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=50)),
                ('current_time', models.CharField(max_length=30)),
                ('current_price', models.CharField(max_length=30)),
                ('price_trend', models.CharField(max_length=10)),
                ('price_volatility', models.CharField(max_length=10)),
                ('click_trend', models.CharField(max_length=10)),
                ('predict_time', models.CharField(max_length=30)),
                ('price_increased', models.CharField(max_length=30)),
                ('real_price', models.CharField(max_length=30)),
            ],
        ),
        migrations.CreateModel(
            name='KNN',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=30)),
                ('time', models.CharField(max_length=30)),
                ('index', models.CharField(max_length=30)),
            ],
        ),
        migrations.CreateModel(
            name='Logistic',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=30)),
                ('time', models.CharField(max_length=30)),
                ('index', models.CharField(max_length=30)),
            ],
        ),
        migrations.CreateModel(
            name='Market',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('time', models.CharField(max_length=30)),
                ('kosdaq_index', models.CharField(max_length=30)),
                ('kospi_index', models.CharField(max_length=30)),
            ],
        ),
        migrations.CreateModel(
            name='PCR',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=30)),
                ('time', models.CharField(max_length=30)),
                ('index', models.CharField(max_length=30)),
            ],
        ),
        migrations.CreateModel(
            name='PLS',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=30)),
                ('time', models.CharField(max_length=30)),
                ('index', models.CharField(max_length=30)),
            ],
        ),
        migrations.CreateModel(
            name='RandomForest',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=30)),
                ('time', models.CharField(max_length=30)),
                ('index', models.CharField(max_length=30)),
            ],
        ),
        migrations.CreateModel(
            name='Result',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=30)),
                ('num_trade', models.CharField(max_length=30)),
                ('percent', models.CharField(max_length=30)),
                ('expected', models.CharField(max_length=30)),
            ],
        ),
        migrations.CreateModel(
            name='Transaction',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=30)),
                ('search_time', models.CharField(max_length=30)),
                ('selling_time', models.CharField(max_length=30)),
                ('buying_price', models.CharField(max_length=30)),
                ('actual_price', models.CharField(max_length=30)),
                ('price_increase', models.CharField(max_length=30)),
            ],
        ),
        migrations.CreateModel(
            name='Tree',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=30)),
                ('time', models.CharField(max_length=30)),
                ('index', models.CharField(max_length=30)),
            ],
        ),
    ]
