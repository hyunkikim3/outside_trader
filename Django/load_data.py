import json, sys, os
import pandas as pd
import numpy as np

project_dir = "~/Django/mysite"

sys.path.append(project_dir)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mysite.settings")

import django

django.setup()

from main.models import Data, Market, KNN, PLS, Logistic, RandomForest, Bagging, Boosting, PCR, Tree, Transaction, Result

with open('final_full.json') as f:
	full_data = json.load(f)

COLUMNS = ['index','name', 'code', 'time', 'price', 'time_1', 'price_1', 'price_dif_1', 'sell_1', 
'buy_1', 'volume_1', 'variation_1', 'post_num_1', 'unique_id_1', 'click_1', 'like_1', 
'dislike_1', 'time_2', 'price_2', 'price_dif_2', 'sell_2', 'buy_2', 'volume_2', 'variation_2', 
'post_num_2', 'unique_id_2', 'click_2', 'like_2', 'dislike_2', 'time_3', 'price_3', 
'price_dif_3', 'sell_3', 'buy_3', 'volume_3', 'variation_3', 'post_num_3', 'unique_id_3', 
'click_3', 'like_3', 'dislike_3', 'mkt_cap', 'kospi', 'kosdaq', 'trash', 
'yesterday_closing_price', 'is_maximum', 'is_minimum', 'price_volatility', 'price_trend', 
'average_price_volatility', 'sell_minus_buy_1', 'sell_minus_buy_2', 'sell_minus_buy_3', 
'is_price_gap_stable', 'price_gap_volatility', 'is_like_higher', 'volume_trend', 
'post_num_trend', 'unique_id_trend', 'click_trend', 'price_increase', 'did_price_increase', 
'did_price_033', 'did_price_100', 'did_price_150', 'kospi_ind', 'kosdaq_ind', 'time_slot', 
'ko_inter', 'early_mor', 'morning', 'lunch', 'afternoon', 'late', 'mkt_change', 'alpha', 
'per_now', 'kospi_1', 'kospi_2', 'kospi_3', 'kospi_answer', 'kosdaq_1', 'kosdaq_2', 'kosdaq_3', 
'kosdaq_answer', 'kospi_trend', 'kosdaq_trend', 'kospi_increase', 'kosdaq_increase', 
'market_increase', 'did_opening_price_increase', 'price_1_sq', 'price_dif_1_sq', 'sell_1_sq', 
'buy_1_sq', 'volume_1_sq', 'variation_1_sq', 'post_num_1_sq', 'unique_id_1_sq', 'click_1_sq', 
'like_1_sq', 'dislike_1_sq', 'price_2_sq', 'price_dif_2_sq', 'sell_2_sq', 'buy_2_sq', 
'volume_2_sq', 'variation_2_sq', 'post_num_2_sq', 'unique_id_2_sq', 'click_2_sq', 'like_2_sq', 
'dislike_2_sq', 'price_3_sq', 'price_dif_3_sq', 'sell_3_sq', 'buy_3_sq', 'volume_3_sq', 
'variation_3_sq', 'post_num_3_sq', 'unique_id_3_sq', 'click_3_sq', 'like_3_sq', 'dislike_3_sq', 
'mkt_cap_sq', 'yesterday_closing_price_sq', 'price_volatility_sq', 'price_trend_sq', 
'average_price_volatility_sq', 'sell_minus_buy_1_sq', 'sell_minus_buy_2_sq', 'sell_minus_buy_3_sq', 
'price_gap_volatility_sq', 'volume_trend_sq', 'post_num_trend_sq', 'unique_id_trend_sq', 
'click_trend_sq', 'kospi_ind_sq', 'kosdaq_ind_sq', 'time_slot_sq', 'ko_inter_sq', 'mkt_change_sq', 
'alpha_sq', 'per_now_sq', 'kospi_1_sq', 'kospi_2_sq', 'kospi_3_sq', 'kosdaq_1_sq', 'kosdaq_2_sq', 
'kosdaq_3_sq', 'kospi_trend_sq', 'kosdaq_trend_sq', 'KNN', 'PLS', 'Logistic', 'Random Forest', 'Bagging',
'Boosting', 'PCR', 'Tree']
       
full_df = pd.DataFrame(full_data, columns = COLUMNS)

full_dict = full_df.T.to_dict('list')


for k,v in full_dict.items():
	data = Data()
	data.name = v[1]
	data.current_time = v[29] #time_3
	data.current_price = v[30] #price_3
	data.price_volatility = v[48]
	data.price_trend = v[49]
	data.click_trend = v[60]
	data.predict_time = v[3]#time
	data.price_increased = v[61]
	data.real_price = v[4]#price

	data.save()

###################################################


with open('final_balance.json') as f:
	balance_data = json.load(f)

COLUMNS = ['index', 'KOSPI_balance', 'KOSDAQ_balance', 'KNN_balance', 'PLS_balance', 
'Logistic_balance', 'Random Forest_balance', 'Bagging_balance', 'Boosting_balance', 
'PCR_balance', 'Tree_balance']

balance_df = pd.DataFrame(balance_data, columns = COLUMNS)

balance_dict = balance_df.T.to_dict('list')


for k,v in balance_dict.items():
	market = Market()
	market.time = v[0]
	market.kosdaq_index = v[2] #kosdaq_answer
	market.kospi_index = v[1] #kospi_answer

	market.save()


####################################################

# KNN


for k,v in balance_dict.items():
	knn = KNN()
	knn.name = "KNN"
	knn.time = v[0]
	knn.index = v[3]

	knn.save()

# PLS
for k,v in balance_dict.items():
	pls = PLS()
	pls.name = "PLS"
	pls.time = v[0]
	pls.index = v[4]

	pls.save()


# Logistic
for k,v in balance_dict.items():
	log = Logistic()
	log.name = "Logistic"
	log.time = v[0]
	log.index = v[5]

	log.save()


# Random Forest
for k,v in balance_dict.items():
	rf = RandomForest()
	rf.name = "Random Forest"
	rf.time = v[0]
	rf.index = v[6]

	rf.save()


# Bagging
for k,v in balance_dict.items():
	bag = Bagging()
	bag.name = "Bagging"
	bag.time = v[0]
	bag.index = v[7]

	bag.save()


# Boosting
for k,v in balance_dict.items():
	boost = Boosting()
	boost.name = "Boosting"
	boost.time = v[0]
	boost.index = v[8]

	boost.save()
	

# PCR
for k,v in balance_dict.items():
	pcr = PCR()
	pcr.name = "PCR"
	pcr.time = v[0]
	pcr.index = v[9]

	pcr.save()

# Tree

for k,v in balance_dict.items():
	tree = Tree()
	tree.name = "Tree"
	tree.time = v[0]
	tree.index = v[10]

	tree.save()


###################################################

with open('tree_search_time.json') as f:
	search_data = json.load(f)

COLUMNS = ['time', 'name', 'price_3', 'price', 'price_increase', 'search_time']

search_df = pd.DataFrame(search_data, columns = COLUMNS)

search_dict = search_df.T.to_dict('list')


for k,v in search_dict.items():
	transaction = Transaction()
	transaction.selling_time = v[0]
	transaction.name = v[1]
	transaction.buying_price = v[2]
	transaction.actual_price = v[3]
	transaction.price_increase = v[4]
	transaction.search_time = v[5]

	transaction.save()



def find_result(col):
	num_trade = len(testing[testing[col] == 1])
	percent = testing[testing[col] == 1]['did_price_033'].mean()
	exp = testing[testing[col] == 1]['price_increase'].mean()

	return col, num_trade, percent, exp


testing = full_df[11629:]
COLUMNS = ['model', 'num_trade', 'percent', 'expected_return'] 

mod_col = ['KNN', 'PLS', 'Logistic', 'Random Forest', 'Bagging', 'Boosting', 'PCR', 'Tree']
result_df = pd.DataFrame('NONE', index = range(8), columns=COLUMNS)
result_data = []
for index, row in result_df.iterrows():
	col = mod_col[index]
	col, num_trade, percent, exp = find_result(col)
	result_df.set_value(index, 'model', col)
	result_df.set_value(index, 'num_trade', num_trade)
	result_df.set_value(index, 'percent', percent)
	result_df.set_value(index, 'expected_return', exp)




result_dict = result_df.T.to_dict('list')

for k,v in result_dict.items():
	result = Result()
	result.name = v[0]
	result.num_trade = v[1]
	result.percent = v[2]
	result.expected = v[3]


	result.save()













