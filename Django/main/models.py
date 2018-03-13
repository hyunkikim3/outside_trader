from django.db import models
import html.parser as htmlparser

'''
class Discussion_info(models.Model):
	post_num = models.CharField(max_length=10)
	unique_id = models.CharField(max_length=10)
	click = models.CharField(max_length=10)
	like = models.CharField(max_length=10)
	dislike = models.CharField(max_length=10)
	name = models.CharField(max_length=250, primary_key=True)
	time = models.CharField(max_length=30)

	def __str__(self):
		return str(self.name)
'''


class Data(models.Model):
	name = models.CharField(max_length=50)
	current_time = models.CharField(max_length=30)
	current_price = models.CharField(max_length=30)
	price_trend = models.CharField(max_length=10)
	price_volatility = models.CharField(max_length=10)
	click_trend = models.CharField(max_length=10)
	predict_time = models.CharField(max_length=30)
	price_increased = models.CharField(max_length=30)
	real_price = models.CharField(max_length=30)

	def __str__(self):
		return str(self.name)


class Market(models.Model):
	time = models.CharField(max_length=30)
	kosdaq_index = models.CharField(max_length=30)
	kospi_index = models.CharField(max_length=30)


	def __str__(self):
		return str(self.time)


class KNN(models.Model):
	name = models.CharField(max_length=30)
	time = models.CharField(max_length=30)
	index = models.CharField(max_length=30)


	def timestamp(self):
		return self.time.strftime("%Y-%m-%d %H:%M")

	def __str__(self):
		return str(self.time)


class PLS(models.Model):
	name = models.CharField(max_length=30)
	time = models.CharField(max_length=30)
	index = models.CharField(max_length=30)


	def timestamp(self):
		return self.time.strftime("%Y-%m-%d %H:%M")

	def __str__(self):
		return str(self.time)

class Logistic(models.Model):
	name = models.CharField(max_length=30)
	time = models.CharField(max_length=30)
	index = models.CharField(max_length=30)


	def timestamp(self):
		return self.time.strftime("%Y-%m-%d %H:%M")

	def __str__(self):
		return str(self.time)

class RandomForest(models.Model):
	name = models.CharField(max_length=30)
	time = models.CharField(max_length=30)
	index = models.CharField(max_length=30)
	


	def timestamp(self):
		return self.time.strftime("%Y-%m-%d %H:%M")

	def __str__(self):
		return str(self.time)

class Bagging(models.Model):
	name = models.CharField(max_length=30)
	time = models.CharField(max_length=30)
	index = models.CharField(max_length=30)
	

	def timestamp(self):
		return self.time.strftime("%Y-%m-%d %H:%M")

	def __str__(self):
		return str(self.time)

class Boosting(models.Model):
	name = models.CharField(max_length=30)
	time = models.CharField(max_length=30)
	index = models.CharField(max_length=30)
	
	def timestamp(self):
		return self.time.strftime("%Y-%m-%d %H:%M")

	def __str__(self):
		return str(self.time)

class PCR(models.Model):
	name = models.CharField(max_length=30)
	time = models.CharField(max_length=30)
	index = models.CharField(max_length=30)
	


	def timestamp(self):
		return self.time.strftime("%Y-%m-%d %H:%M")

	def __str__(self):
		return str(self.time)

class Tree(models.Model):
	name = models.CharField(max_length=30)
	time = models.CharField(max_length=30)
	index = models.CharField(max_length=30)
	
	def timestamp(self):
		return self.time.strftime("%Y-%m-%d %H:%M")

	def __str__(self):
		return str(self.time)


class Transaction(models.Model):
	name = models.CharField(max_length=30)
	search_time = models.CharField(max_length=30)
	selling_time = models.CharField(max_length=30)
	buying_price = models.CharField(max_length=30)
	actual_price = models.CharField(max_length=30)
	price_increase = models.CharField(max_length=30)


	def __str__(self):
		return str(self.name)



class Result(models.Model):
	name = models.CharField(max_length=30)
	num_trade = models.CharField(max_length=30)
	percent = models.CharField(max_length=30)
	expected = models.CharField(max_length=30)







