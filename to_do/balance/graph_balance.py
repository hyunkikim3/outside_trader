#run this file to graph results for mkt average and different models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

with open('balance_Feb27_Mar07_final.json', 'r', encoding='UTF-8') as f:
    BALANCE = json.load(f)

with open('model_df_final.json', 'r', encoding='UTF-8') as f:
    MODEL = json.load(f)

COL = ['index', 'name', 'code', 'time', 'price', 'time_1', 'price_1', \
       'price_dif_1', 'sell_1', 'buy_1', 'volume_1', 'variation_1', \
       'post_num_1', 'unique_id_1', 'click_1', 'like_1', 'dislike_1', \
       'time_2', 'price_2', 'price_dif_2', 'sell_2', 'buy_2', 'volume_2', \
       'variation_2', 'post_num_2', 'unique_id_2', 'click_2', 'like_2', \
       'dislike_2', 'time_3', 'price_3', 'price_dif_3', 'sell_3', 'buy_3', \
       'volume_3', 'variation_3', 'post_num_3', 'unique_id_3', 'click_3', \
       'like_3', 'dislike_3', 'mkt_cap', 'kospi', 'kosdaq', 'trash', \
       'yesterday_closing_price', 'is_maximum', 'is_minimum', \
       'price_volatility', 'price_trend', 'average_price_volatility', \
       'sell_minus_buy_1', 'sell_minus_buy_2', 'sell_minus_buy_3', \
       'is_price_gap_stable', 'price_gap_volatility', 'is_like_higher', \
       'volume_trend', 'post_num_trend', 'unique_id_trend', 'click_trend', \
       'price_increase', 'did_price_increase', 'did_price_033', \
       'did_price_100', 'did_price_150', 'kospi_ind', 'kosdaq_ind', \
       'time_slot', 'ko_inter', 'early_mor', 'morning', 'lunch', \
       'afternoon', 'late', 'mkt_change', 'alpha', 'per_now', 'kospi_1', \
       'kospi_2', 'kospi_3', 'kospi_answer', 'kosdaq_1', 'kosdaq_2', \
       'kosdaq_3', 'kosdaq_answer', 'kospi_trend', 'kosdaq_trend', \
       'kospi_increase', 'kosdaq_increase', 'market_increase', \
       'did_opening_price_increase', 'price_1_sq', 'price_dif_1_sq', \
       'sell_1_sq', 'buy_1_sq', 'volume_1_sq', 'variation_1_sq', \
       'post_num_1_sq', 'unique_id_1_sq', 'click_1_sq', 'like_1_sq', \
       'dislike_1_sq', 'price_2_sq', 'price_dif_2_sq', 'sell_2_sq', \
       'buy_2_sq', 'volume_2_sq', 'variation_2_sq', 'post_num_2_sq', \
       'unique_id_2_sq', 'click_2_sq', 'like_2_sq', 'dislike_2_sq', \
       'price_3_sq', 'price_dif_3_sq', 'sell_3_sq', 'buy_3_sq', \
       'volume_3_sq', 'variation_3_sq', 'post_num_3_sq', 'unique_id_3_sq', \
       'click_3_sq', 'like_3_sq', 'dislike_3_sq', 'mkt_cap_sq', \
       'yesterday_closing_price_sq', 'price_volatility_sq', \
       'price_trend_sq', 'average_price_volatility_sq', \
       'sell_minus_buy_1_sq', 'sell_minus_buy_2_sq', 'sell_minus_buy_3_sq', \
       'price_gap_volatility_sq', 'volume_trend_sq', 'post_num_trend_sq', \
       'unique_id_trend_sq', 'click_trend_sq', 'kospi_ind_sq', \
       'kosdaq_ind_sq', 'time_slot_sq', 'ko_inter_sq', 'mkt_change_sq', \
       'alpha_sq', 'per_now_sq', 'kospi_1_sq', 'kospi_2_sq', 'kospi_3_sq', \
       'kosdaq_1_sq', 'kosdaq_2_sq', 'kosdaq_3_sq', 'kospi_trend_sq', \
       'kosdaq_trend_sq', 'KNN', 'PLS', 'Logistic', 'Random Forest', \
       'Bagging', 'Boosting', 'PCR', 'Tree']

METHOD = ['AVG', 'KNN', 'PLS', 'Logistic', 'Random Forest', 'Bagging', \
          'Boosting', 'PCR', 'Tree']
BALANCE_COL = ["index", "KOSPI_balance", "KOSDAQ_balance", "KNN_balance", \
               "PLS_balance", "Logistic_balance", "Random Forest_balance", \
               "Bagging_balance", "Boosting_balance", "PCR_balance", \
               "Tree_balance"]

TOTAL = pd.DataFrame(MODEL, columns=COL)
RESULT = pd.DataFrame(BALANCE, columns=BALANCE_COL)
TESTING = TOTAL[11629:]


def draw_result(model):
    '''
    Draw graph and print results of the specific model and market result.

    Input:
      model: the string of model to be drawn, if 'AVG', only grphs mkt
             indices

    Return:
    '''
    
  
    if model == 'AVG':
        df = pd.DataFrame({'x': range(0, 146), \
        	               'y1': RESULT["KOSPI_balance"], \
        	               'y2': RESULT["KOSDAQ_balance"]})
        sub_df = TESTING
        plt.title('average')
    else:
        balance_text = balance_text = model + '_balance'
        df = pd.DataFrame({'x': range(0, 146), \
        	               'y1': RESULT["KOSPI_balance"], \
        	               'y2': RESULT["KOSDAQ_balance"],\
                           'y3': RESULT[balance_text]})
        sub_df = TESTING[TESTING[model] == 1]
        plt.title(model)
        plt.plot(df['x'], df['y3'], label=model)
        print("Number of trades:", len(sub_df))

    plt.plot(df['x'], df['y1'], label="KOSPI")
    plt.plot(df['x'], df['y2'], label="KOSDAQ")
    plt.legend()
    plt.show()    
    print("Proportion increased more than 0.33%:", \
    	  sub_df["did_price_033"].mean())
    print("Expected return:", sub_df["price_increase"].mean())
    
    return

for method in METHOD:
	draw_result(method)


