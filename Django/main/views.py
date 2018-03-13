from django.shortcuts import render
from django.http import HttpResponse
from . models import Data, Market, KNN, PLS, Logistic, RandomForest, Bagging, Boosting, PCR, Tree, Transaction, Result
from django.db.models import Max



def admin(request):
	return render(request, 'mysite/main.html')



def index(request):
	return render(request, 'main/header.html')


def choose(request):

	return render(request, 'main/choose.html')



def result(request):


	find_time = request.GET.get('find_time')

	find_data = Data.objects.filter(current_time = find_time)

	context ={

			'find_time' : find_time,
			'find_data' : find_data,
	}

	return render(request, 'main/result.html', context)


def performance(request):

	result = Result.objects.all()

	context = {
		'result' : result
	}

	return render(request, 'main/performance.html', context)

	

def best(request):

	maximum = Transaction.objects.all().aggregate(Max('price_increase'))

	context = {
		'maximum' : maximum,
	}

	return render(request, 'main/best.html', context)


def best_table(request):

	trans = Transaction.objects.all()

	context = {
		'trans' : trans
	}

	return render(request, 'main/best_table.html', context)


def graph(request):

	market = Market.objects.all()
	data = Data.objects.all()


	context = {
		'market' : market,
		'data' : data,
	}

	return render(request, 'main/graph.html', context)


