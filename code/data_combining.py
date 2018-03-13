import numpy as np
import pandas as pd
import re
import json

try:
    with open("../data/krx_code.json", "r", encoding="UTF-8") as f:
        KRX_CODE = json.load(f)
except FileNotFoundError as e:
    print(e)

try:
    with open("../data/company_info.json", 'r', encoding='UTF-8') as f:
        COMPANY_INFO = json.load(f)
except FileNotFoundError as e:
    print(e)
    
kospi = {}
kosdaq = {}
for month in range(2, 4):
    for day in range(1, 32):
        prefix = "../data/market/2018-" + ("0" + str(month) if month <= 9 else str(month)) +\
                   "-" + ("0" + str(day) if day <= 9 else str(day)) + "_market/"
        for market in ("KOSPI", "KOSDAQ"):
            filename = prefix + market + "_2018-" + ("0" + str(month) if month <= 9 else str(month)) +\
                       "-" + ("0" + str(day) if day <= 9 else str(day)) + ".json"
            
            try:
                with open(filename, 'r', encoding='UTF-8') as f:
                    opened = json.load(f)
                    
                    if market == "KOSPI":
                        kospi.update(opened)
                    else:
                        kosdaq.update(opened)

            except FileNotFoundError:
                continue

company_df = pd.DataFrame(COMPANY_INFO, columns = ["name", "code", \
                                                   "market", "size"])
KOSPI = []
KOSDAQ = []
TRASH = []
MKT_CAP = {}

for idx, row in company_df.iterrows():
    if re.sub('[0-9 ,위]', '', row["market"]) == "코스피":
        KOSPI.append(row["code"])
    elif re.sub('[0-9 ,위]', '', row["market"]) == "코스닥":
        KOSDAQ.append(row["code"])
    else:
        TRASH.append(row["code"])    
    MKT_CAP[row["code"]] = int(re.sub('[조억원,]', '', row["size"]))


COLUMN_DISC =["name", "time", "post_num", "unique_id", "click", \
              "like", "dislike", "time_1", "post_num_1", "unique_id_1", \
              "click_1", "like_1", "dislike_1", "time_2", "post_num_2", \
              "unique_id_2", "click_2", "like_2", "dislike_2", "time_3", \
              "post_num_3", "unique_id_3", "click_3", "like_3", "dislike_3"]

COLUMN_PRICE=["code", "name", "time", "price", "price_dif", "sell", "buy", \
              "volume", "variation", "time_1", "price_1", "price_dif_1", \
              "sell_1", "buy_1", "volume_1", "variation_1", "time_2", \
              "price_2", "price_dif_2", "sell_2", "buy_2", "volume_2", \
              "variation_2", "time_3", "price_3", "price_dif_3", "sell_3", \
              "buy_3", "volume_3", "variation_3"]

COLUMN_TOTAL = ["name", "code", "time", "price", "time_1", "price_1", \
                "price_dif_1", "sell_1", "buy_1", "volume_1", \
                "variation_1", "post_num_1", "unique_id_1", "click_1", \
                "like_1", "dislike_1", "time_2", "price_2", "price_dif_2", \
                "sell_2", "buy_2", "volume_2", "variation_2", "post_num_2", \
                "unique_id_2", "click_2", "like_2", "dislike_2", "time_3", \
                "price_3", "price_dif_3", "sell_3", "buy_3", "volume_3", \
                "variation_3", "post_num_3", "unique_id_3", "click_3", \
                "like_3", "dislike_3"]

VAR_TO_TRANSFORM = ['price', 'price_1', 'price_dif_1', 'sell_1', 'buy_1', \
                    'volume_1', 'variation_1', 'price_2', 'price_dif_2', \
                    'sell_2', 'buy_2', 'volume_2',  'variation_2', \
                    'price_3', 'price_dif_3', 'sell_3', 'buy_3', 'volume_3', \
                    'variation_3']

MISSING = ["2018-02-27 11:30", "2018-02-27 11:40", "2018-02-27 11:50", \
           "2018-02-27 12:00", "2018-02-27 12:10", "2018-02-27 12:20", \
           "2018-02-27 12:30", "2018-02-27 12:40", "2018-02-27 12:50", \
           "2018-02-27 13:00", "2018-02-27 13:10", "2018-02-27 13:20", \
           "2018-02-27 13:30", "2018-02-27 13:40"]

PRICE_SINGLE_COL = ["code", "name", "time", "price", "price_dif", \
                    "sell", "buy", "volume", "variation"]

SQUARED = ['price_1', 'price_dif_1', 'sell_1', 'buy_1', 'volume_1', \
         'variation_1', 'post_num_1', 'unique_id_1', 'click_1', 'like_1', \
         'dislike_1', 'price_2', 'price_dif_2', 'sell_2', 'buy_2', \
         'volume_2', 'variation_2', 'post_num_2', 'unique_id_2', 'click_2', \
         'like_2', 'dislike_2', 'price_3', 'price_dif_3', 'sell_3', 'buy_3', \
         'volume_3', 'variation_3', 'post_num_3', 'unique_id_3', 'click_3', \
         'like_3', 'dislike_3', 'mkt_cap', 'yesterday_closing_price', \
         'price_volatility', 'price_trend', 'average_price_volatility', \
         'sell_minus_buy_1', 'sell_minus_buy_2', 'sell_minus_buy_3', \
         'price_gap_volatility', 'volume_trend', 'post_num_trend', \
         'unique_id_trend', 'click_trend', 'kospi_ind', 'kosdaq_ind', \
         'time_slot', 'ko_inter', 'mkt_change', 'alpha', 'per_now', \
         'kospi_1', 'kospi_2', 'kospi_3', 'kosdaq_1', 'kosdaq_2', \
         'kosdaq_3', 'kospi_trend', 'kosdaq_trend']


def list_to_df(date, df_list, column_names, key_list):
    ''''
    Transform a list of dataframes into a whole dataframe
    we want, in each row recording information of price 
    and discussion of each stock and the these information
    60 minutes ago, 70 minutes ago and 80 minutes ago.    
    
    Inputs:
      date: string of date, e.g. '2018-03-06'
      df_list: a list of dataframe
      column_names: list of columns
      key_list: list of columns to merge data frames
      
    Return: a dataframe
    '''
    total = pd.DataFrame(columns=column_names)
    for ind, df in enumerate(df_list):
        
        if ind >= 8:
            df_total = df.merge(df_list[ind - 8], on = \
                                key_list).merge(df_list[ind - 7], \
                                on = key_list).merge(df_list[ind - 6], \
                                on = key_list)
            df_total.columns = column_names
            total = pd.concat([total, df_total], axis = 0)

    return total


def get_price(date):
    '''
    Transform all the raw price files on a given date into a whole dataframe.
    
    Input:
      date: string of date, e.g. '2018-03-06'
      
    Return: a dataframe
    '''
    price_text = "../data/price/" + date + "_price/" +\
                 date + "_price.json"
    
    with open(price_text, 'r', encoding='UTF-8') as f:
        price = json.load(f)
    
    price_df = pd.DataFrame(price, columns = ["index", "code", "name", \
                                              "time", "price", "price_dif", \
                                              "sell", "buy", "volume", \
                                              "variation"])
    price_df = price_df[PRICE_SINGLE_COL][1:]
    text = date + " 09:00"
    price_df = price_df[price_df["time"] != text]
    time_list = price_df['time'].unique().tolist()
    
    price_df_list = []
    for time in time_list:
        df = price_df[price_df["time"] == time]
        df = df[PRICE_SINGLE_COL]
        price_df_list.append(df)
    
    price_df = list_to_df(date, price_df_list, COLUMN_PRICE, \
                          ['code', 'name']).reset_index().drop(["index"], axis = 1)
    
    return price_df


def get_discussion(date):
    '''
    Transform all the raw price files on a given date into a whole dataframe.
    
    Input:
      date: string of date, e.g. '2018-03-06'
      
    Return: a dataframe
    '''
    
    df_list = []
    
    time = []
    prefix = "../data/discussion/" + date + "_focus/discussion_" + date
    month = date[6]
    day = date[8:]
    if month == "2":
        for hour in range(9,16):        
            if day == "26" and hour == 9:
                continue        
            elif day == "27" and hour == 11:
                continue            
            for minute in range(0, 6):
                if hour == 15 and minute > 3:
                    break
                if day == "26" and hour == 10 and minute == 0:
                    continue            
                elif day == "27" and hour == 10 and minute > 2:
                    continue            
                elif day == "27" and hour == 12 and minute < 3:
                    continue            
                time.append(prefix + "-" + (("0" + str(hour)) if hour <= 9 \
                            else str(hour)) + "-" + str(minute) + "0.json")
    
    elif month == "3":  
        for hour in range(9,16):    
            if day == "02" and hour == 9:
                continue        
            elif day == "02" and hour == 10:
                continue        
            elif day == "02" and hour == 15:
                continue            
            for minute in range(0, 6):
                if hour == 15 and minute > 3:
                    break
                if day == "02" and hour == 11 and minute < 5:
                    continue            
                elif day == "02" and hour == 14 and minute > 4:
                    continue            
                time.append(prefix + "-" + (("0" + str(hour)) if hour <= 9 \
                            else str(hour)) + "-" + str(minute) + "0.json")
    
    for x in time:
        with open(x, 'r', encoding='UTF-8') as f:
            discussion = json.load(f)        
            discuss_df = pd.DataFrame(discussion, columns = ["post_num", \
                                "unique_id", "click", "like", "dislike", 
                                "name", "time"])
            reset_col = ["name", "time", "post_num", "unique_id", "click", \
                         "like", "dislike"]
            discuss_df = discuss_df[reset_col]
            df_list.append(discuss_df)
    
    discuss_df = list_to_df(date, df_list, COLUMN_DISC, 
                            ['name']).reset_index().drop(["index"], \
                            axis = 1)

    return discuss_df


def daily_dataframe(date):
    '''
    Add company info to the whole dataframe created from the price and 
    discussion raw data.
    
    Input:
      date: string of date, e.g. '2018-03-06'
      
    Return: a dataframe
    '''
    price_df = get_price(date)
    discuss_df = get_discussion(date)
    total = pd.merge(price_df, discuss_df, on = ['name', \
                                                    'time', 'time_1', \
                                                    'time_2', 'time_3'])
    total = total[COLUMN_TOTAL]
    
    total["mkt_cap"] = np.nan
    total["kospi"] = np.nan
    total["kosdaq"] = np.nan
    total["trash"] = np.nan
    
    for index, row in total.iterrows():
        mkt_cap = MKT_CAP[row["code"]]

        kospi_dummy = 1 if row["code"] in KOSPI else 0

        kosdaq_dummy = 1 if row["code"] in KOSDAQ else 0

        trash = 1 if row["code"] in TRASH else 0

        total.set_value(index,"mkt_cap", mkt_cap)
        total.set_value(index,"kospi", kospi_dummy)
        total.set_value(index,"kosdaq", kosdaq_dummy)
        total.set_value(index,"trash", trash)
    
    total.dropna(inplace = True)
    for var in VAR_TO_TRANSFORM:
        total = total[total[var] != '\xa0']
        for index, row in total.iterrows():
            if isinstance(row[var], str):
                data = row[var].split(",")
                value = ''.join(data)
                value = int(value)
                total.set_value(index, var, value)

    return total


def complete_dataframe(dates):
    '''
    Make a complete dataframe with modified variables ready for analyze.
    
    Input:
      dates: a list of dates to mark the data
      
    Return: a dataframe     
    '''
    total_df = pd.DataFrame(columns=COLUMN_TOTAL)
    for date in dates:
        df = daily_dataframe(date)
        df = df[~df['time'].isin(MISSING)]
        total_df = pd.concat([total_df, df])
    total_df = total_df.reset_index().drop(["index"], axis = 1)
    
    total_df["yesterday_closing_price"] = total_df["price_1"] - \
                                          total_df["price_dif_1"]
    total_df["is_maximum"] = (((total_df["price_1"] / \
                             total_df["yesterday_closing_price"]) - 1) * \
                             100 > 29.5) | (((total_df["price_2"] / \
                             total_df["yesterday_closing_price"]) - 1) * \
                             100 > 29.5) | (((total_df["price_3"] / \
                             total_df["yesterday_closing_price"]) - 1) * \
                             100 > 29.5)
    total_df["is_maximum"] = total_df["is_maximum"].astype(int)

    total_df["is_minimum"] = (((total_df["price_1"] / \
                             total_df["yesterday_closing_price"]) - 1) * \
                             100 < -29.5) | (((total_df["price_2"] / \
                             total_df["yesterday_closing_price"]) - 1) * \
                             100 < -29.5) | (((total_df["price_3"] / \
                             total_df["yesterday_closing_price"]) - 1) * \
                             100 < -29.5)
    total_df["is_minimum"] = total_df["is_maximum"].astype(int)

    total_df["price_volatility"] = (((total_df[["price_1", "price_2", \
                                   "price_3"]].max(axis=1)) / \
                                   (total_df[["price_1", "price_2", \
                                    "price_3"]].min(axis=1))) - 1) * 100

    total_df["price_trend"] = (((total_df["price_1"] - \
                              total_df["price_2"]) < 0).astype(int) + \
                              ((total_df["price_2"] - total_df["price_3"]) < \
                              0).astype(int)) - (((total_df["price_1"] - \
                              total_df["price_2"]) > 0).astype(int) + \
                              ((total_df["price_2"] - total_df["price_3"]) > \
                              0).astype(int))

    total_df["average_price_volatility"] = total_df["price_trend"] * \
                                           total_df["price_volatility"] / 2
    
    for i in range(1, 4):
        minus, sell, buy = "sell_minus_buy_" + str(i), "sell_" + str(i), \
                           "buy_" + str(i)
        total_df[minus] = total_df[sell] - total_df[buy]

    total_df["is_price_gap_stable"] = ((total_df["sell_minus_buy_1"] == \
                                      total_df["sell_minus_buy_2"]) & \
                                      (total_df["sell_minus_buy_2"] == \
                                   total_df["sell_minus_buy_3"])).astype(int)

    total_df["price_gap_volatility"] = (((total_df[["sell_minus_buy_1", \
                                       "sell_minus_buy_2", 
                                       "sell_minus_buy_3"]].max(axis=1)) / \
                                       (total_df[["sell_minus_buy_1", \
                                       "sell_minus_buy_2", \
                                       "sell_minus_buy_3"]].min(axis=1)))) - 1

    total_df["is_like_higher"] = (total_df["like_3"] > \
                                 total_df["dislike_3"]).astype(int)
    
    disc_trend = ["volume", "post_num", "unique_id", "click"]
    for var in disc_trend:
        var_trend = var + "_trend"
        var_3, var_2, var_1 = var + "_3", var + "_2", var + "_1"
        total_df[var_trend] = ((((total_df[var_3]) - \
                              (total_df[var_2] * 1+1e-3)) / \
                              ((total_df[var_2]) - (total_df[var_1] * \
                              1-1e-4))) - 1) * 100

    total_df["price_increase"] = ((total_df["price"] / total_df["price_3"]) \
                                   - 1) * 100

    total_df["did_price_increase"] = (total_df["price_increase"] > \
                                      0).astype(int)

    total_df["did_price_033"] = (total_df["price_increase"] > \
                                 0.33).astype(int)

    total_df["did_price_100"] = (total_df["price_increase"] > 1.0).astype(int)

    total_df["did_price_150"] = (total_df["price_increase"] > 1.5).astype(int)
    
    mkt_time_new = ["kospi_ind", "kosdaq_ind", "time_slot", "ko_inter", \
                    "early_mor", "morning", "lunch", "afternoon", "late", \
                    "mkt_change", "alpha", "per_now", "kospi_1", "kospi_2", \
                    "kospi_3", "kospi_answer", "kosdaq_1", "kosdaq_2", \
                    "kosdaq_3", "kosdaq_answer"]
    for var in mkt_time_new:
        total_df[var] = np.nan
    num = 1
    time_list = []
    for index, row in total_df.iterrows():
        kospi_ind = kospi[row['time']]
        kosdaq_ind = kosdaq[row['time']]
        total_df.set_value(index,'kospi_ind', kospi_ind)
        total_df.set_value(index,'kosdaq_ind', kosdaq_ind)
        time = row['time'].split()[1]
        if time not in time_list:
            time_list.append(time)
        ind = time_list.index(time)
        num = index + 1

        early_mor = 1 if num in [1, 2, 3] else 0

        morning = 1 if (num >= 1) and (num <= 15) else 0

        lunch = 1 if (num >= 16) and (num <= 24) else 0
    
        afternoon = 1 if (num >= 25) and (num <= 36) else 0

        late = 1 if (num >= 31) and (num <= 36) else 0

        mkt_change = row['kospi'] * kospi_ind + row['kosdaq'] * kosdaq_ind
        total_df.set_value(index,'mkt_change', mkt_change)
        total_df.set_value(index,'early_mor', early_mor)
        total_df.set_value(index,'morning', morning)
        total_df.set_value(index,'lunch', lunch)
        total_df.set_value(index,'afternoon', afternoon)
        total_df.set_value(index,'late', late)
        total_df.set_value(index,'time_slot', num)
        total_df.set_value(index,'ko_inter', kospi_ind * kosdaq_ind)
        per_now = 100 * row['price_dif_3']/row['yesterday_closing_price']
        total_df.set_value(index, 'per_now', per_now)
        alpha = per_now - mkt_change
        total_df.set_value(index, 'alpha', alpha)
        
        last_closing = row["time"][:10] + " last_closing"
        
        total_df.set_value(index, 'kospi_1', (kospi[row["time_1"]] / kospi[last_closing] - 1) * 100)
        total_df.set_value(index, 'kospi_2', (kospi[row["time_2"]] / kospi[last_closing] - 1) * 100)
        total_df.set_value(index, 'kospi_3', (kospi[row["time_3"]] / kospi[last_closing] - 1) * 100)
        total_df.set_value(index, 'kospi_answer', kospi[row["time"]])
     
        total_df.set_value(index, 'kosdaq_1', (kosdaq[row["time_1"]] / kosdaq[last_closing] - 1) * 100)
        total_df.set_value(index, 'kosdaq_2', (kosdaq[row["time_2"]] / kosdaq[last_closing] - 1) * 100)
        total_df.set_value(index, 'kosdaq_3', (kosdaq[row["time_3"]] / kosdaq[last_closing] - 1) * 100)
        total_df.set_value(index, 'kosdaq_answer', kosdaq[row["time"]])

    total_df["kospi_trend"] = ((((total_df["kospi_3"]) - \
                              (total_df["kospi_2"] * 1+1e-3)) / \
                              ((total_df["kospi_2"]) - (total_df["kospi_1"] \
                              * 1-1e-4))) - 1) * 100

    total_df["kosdaq_trend"] = ((((total_df["kosdaq_3"]) - \
                               (total_df["kosdaq_2"] * 1+1e-3)) / \
                               ((total_df["kosdaq_2"]) - \
                               (total_df["kosdaq_1"] * 1-1e-4))) - 1) * 100

    total_df["kospi_increase"] = 100 * (total_df["kospi_answer"] - \
                                 total_df["kospi_3"]) / total_df["kospi_3"]

    total_df["kosdaq_increase"] = 100 * (total_df["kosdaq_answer"] - \
                                  total_df["kosdaq_3"]) / total_df["kosdaq_3"]

    total_df["market_increase"] = (total_df["kospi"] * \
                                   total_df["kospi_increase"]) + \
                                  (total_df["kosdaq"] * \
                                   total_df["kosdaq_increase"])
    
    total_df["did_opening_price_increase"] = 1

    for var in SQUARED:
        col_name = var + '_sq'
        total_df[col_name] = np.nan
        for index, row in total_df.iterrows():
            sqr = row[var] ** 2
            total_df.set_value(index, col_name, sqr)
    
    return total_df


def save_monthly_dataframe(month):
    '''
    Save the dataframe of the given month in json file. 
    
    Input: 
      month: integer, 2 or 3
    
    Return a dataframe
    '''
    trading = []
    for day in range(1,32):
        date = "2018-" + ("0" + str(month) if month <= 9 else str(month)) +\
               "-" + ("0" + str(day) if day <= 9 else str(day))
        path = "../data/discussion/" + date + "_focus/" + date + "_focus_group.json"

        try:
            with open(path, "r", encoding="UTF-8"):
                trading.append(date)

        except FileNotFoundError:
            continue

    df = complete_dataframe(trading)
    df.to_json("dataframe_" + ("0" + str(month) if month <= 9 else str(month)) + ".json", orient='values')
    
    return df
