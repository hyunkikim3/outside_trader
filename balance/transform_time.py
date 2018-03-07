import time
import calendar 
import json
import pandas as pd
import numpy as np

COLUMNS = ["time", "KOSPI_balance", "KOSDAQ_balance", "KNN_balance", \
           "PLS_balance", "Logistic_balance", "Random Forest_balance", \
           "Bagging_balance", "Boosting_balance", "PCR_balance", \
           "Tree_balance"]

def transform(filename, col_list):
	'''
	Split balance data of kospi and kosdaq and different models	balance
	into different json files with time as epoch GMT as index. 

	Inputs:
	  filename: the name of the balance json file
	  col_list: the columns in the file to be split

	 Return
	'''

	with open(filename, 'r') as f:
		balance = json.load(f)
	balance_df = pd.DataFrame(balance, columns = COLUMNS)
	time_df = pd.DataFrame(np.nan, index = range(balance_df.shape[0]), columns = ['time'])
	for index, row in balance_df.iterrows():
		timestr = row['time']
		dg_time = calendar.timegm(time.strptime(timestr, "%Y-%m-%d %H:%M")) * 1000
		time_df.set_value(index, 'time', dg_time)
	for col in col_list:
		df = pd.concat([time_df, pd.DataFrame(balance_df[col], columns=[col])], axis = 1)
		filename = col + '.json'
		df.to_json(filename, orient = 'values')

	return 




