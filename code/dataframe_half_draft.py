import scraper

import numpy as np
import pandas as pd
import re
import json
import bs4
import urllib3
from datetime import datetime, tzinfo, timedelta
from pytz import timezone

try:
    with open("../raw_data/krx_code.json", "r", encoding="UTF-8") as krx:
        KRX_CODE = json.load(krx)

except FileNotFoundError as e:
    print(e)


def combine_price(date, save=False):
    '''
    Combine price and discussion info into a dataframe with on the given date
    and save them in a list
    Input:
      date: string, e.g. "2018-03-06"
    Return: a list
    '''
    df = pd.DataFrame(columns=["code", "name", "time", "price", \
                               "price_dif", "sell", "buy", "volume", \
                               "variation"])
    rv = []
    rv.append(df.columns.tolist())

    time = []
    for hour in range(9,16):
        for minute in range(0, 6):
            if hour != 15 or minute < 4:
                time.append("../raw_data/discussion/" + date + \
                "_focus/discussion_" + date + "-" + (("0" + str(hour)) \
                if hour <= 9 else str(hour))+ "-" + str(minute) + "0.json")

    for t in time:
        try:
            with open(t, 'r', encoding='UTF-8') as f:
                discussion = json.load(f)

        except FileNotFoundError as e:
            print(e)
            continue

        opening_increase = "../raw_data/price/" + date + "_price/" + date +\
                            "_opening_increase.json"

        try:
            with open(opening_increase, 'r', encoding='UTF-8') as f:
                increased = json.load(f)

        except FileNotFoundError as e:
            print(e)
            continue


        for d in discussion:
            if d["name"] in increased:
                row = pd.DataFrame(columns=["code", "name", "time", \
                                            "price", "price_dif", "sell", \
                                            "buy", "volume", "variation"], \
                                   data = [[KRX_CODE[d["name"]], d["name"], \
                                            d["time"], np.nan, np.nan,\
                                            np.nan, np.nan, np.nan, np.nan]])
                df = df.append(row)

    df = df.reset_index()
    
    
    for idx, row in df[:4].iterrows():
        timestamp = row["time"]
        t = re.sub('[ :-]', '', timestamp)

        d = scraper.scrape_price_history(row["code"], t)

        df["price"].iloc[idx] = d["price"]
        df["price_dif"].iloc[idx] = d["price_dif"]
        df["sell"].iloc[idx] = d["sell"]
        df["buy"].iloc[idx] = d["buy"]
        df["volume"].iloc[idx] = d["volume"]
        df["variation"].iloc[idx] = d["variation"]
    
    rv = []
    rv.append(df.columns.tolist())
    for row in df.iterrows():
        rv.append(row[1].tolist())

    if save:

        with open(date + "_price.json","w", encoding='UTF-8') as f:
            json.dump(rv, f, ensure_ascii=False)

    return rv



with open("krx_code.json", 'r', encoding='UTF-8') as krx:
    KRX_CODE = json.load(krx)

with open("company_info.json", 'r', encoding='UTF-8') as f:
    COMPANY_INFO = json.load(f)

with open('KOSPI_Feb14_Mar07.json', 'r', encoding='UTF-8') as f:
    KOSPI_DIFF = json.load(f)

with open('KOSDAQ_Feb14_Mar07.json', 'r', encoding='UTF-8') as f:
    KOSDAQ_DIFF = json.load(f)

with open('kospi_now_Feb14_Mar07.json', 'r', encoding='UTF-8') as f:
    KOSPI_NOW = json.load(f)

with open('kosdaq_now_Feb14_Mar07.json', 'r', encoding='UTF-8') as f:
    KOSDAQ_NOW = json.load(f)


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

COLUMN_TOTAL = ['name', 'code', 'time', 'price', 'time_1', "price_1", \
                "price_dif_1", "sell_1", "buy_1", "volume_1", \
                "variation_1", 'post_num_1', 'unique_id_1', 'click_1', \
                'like_1', 'dislike_1', 'time_2', 'price_2', "price_dif_2", \
                "sell_2", "buy_2", "volume_2", "variation_2", 'post_num_2', \
                'unique_id_2', 'click_2', 'like_2', 'dislike_2', 'time_3', \
                'price_3', "price_dif_3", "sell_3", "buy_3", "volume_3", \
                "variation_3", 'post_num_3', 'unique_id_3', 'click_3', \
                'like_3', 'dislike_3']

VAR_TO_TRANSFORM = ['price', 'price_1', 'price_dif_1', 'sell_1', 'buy_1', \
                    'volume_1', 'variation_1', 'price_2', 'price_dif_2', \
                    'sell_2', 'buy_2', 'volume_2',  'variation_2', \
                    'price_3', 'price_dif_3', 'sell_3', 'buy_3', 'volume_3', \
                    'variation_3']

DEL_TIME = ["2018-02-27 11:30", "2018-02-27 11:40", "2018-02-27 11:50", \
            "2018-02-27 12:00", "2018-02-27 12:10", "2018-02-27 12:20", \
            "2018-02-27 12:30", "2018-02-27 12:40", "2018-02-27 12:50", \
            "2018-02-27 13:00", "2018-02-27 13:10", "2018-02-27 13:20", \
            "2018-02-27 13:30", "2018-02-27 13:40"]

PRICE_SINGLE_COL = ["code", "name", "time", "price", "price_dif", \
                    "sell", "buy", "volume", "variation"]

TO_SQ = ['price_1', 'price_dif_1', 'sell_1', 'buy_1', 'volume_1', \
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


def open_files(date):
    ''''
    Open files for making data frame for the specific date
    Input:
      date: string of date, e.g. '2018-03-06', only apply to Feburary 14, 20, 21, 
            22, 23, 26, 27, 28, and March 2, 3, 6 ,7
    Return: dictionary, dictionary
    '''
    
    focus_text = date + "_focus/" + date + "_focus_group.json"

    price_text = date + "_price_and_everything.json"
    
    with open(focus_text, 'r', encoding='UTF-8') as focus_group:
        focus = json.load(focus_group)
    with open(price_text, 'r', encoding='UTF-8') as pnc:
        price = json.load(pnc)
    
    return focus, price


def get_single_time(prefix, hour, minute):
    '''
    Get a string of time based on given prefix, hour and min
    
    Inputs:
      prefix: string of path, e.g. "2018-02-28_focus/discussion_2018-02-28"
      hour: integer of hour, from 9 to 15
      min: integer of min, from 0 to 6
    Return: a string of path
    '''
    
    if hour == 15 and minute > 3:
        return 
    else:
        rv = prefix + " " + (("0" + str(hour)) if hour <= 9 else str(hour)) \
             + ":" + str(minute) + "0.json"

    return rv


def get_time_disc(date):
    '''
    Get the list for discussion data filenames of different time for the 
    specific day, e.g. from 2018-03-06 9 am to 3 pm every ten minutes. 
    Input:
      date: string of date, e.g. 2018-03-06, only apply to Feburary 14, 20, 
            21, 22, 23, 26, 27, 28, and March 2, 3, 6 ,7
    Return: a list
    '''
    
    time = []
    prefix = date + "_focus/discussion_" + date
    month = date[6]
    day = date[8:]
    if month == '2':
        for hour in range(9,16):        
            if day == "26" and hour == 9:
                continue        
            elif day == "27" and hour == 11:
                continue            
            for minute in range(0, 6):
                if day == "26" and hour == 10 and minute == 0:
                    continue            
                elif day == "27" and hour == 10 and minute > 2:
                    continue            
                elif day == "27" and hour == 12 and minute < 3:
                    continue            
                time.append(get_single_time(prefix, hour, minute))            
    elif month == '3':  
        for hour in range(9,16):        
            if day == "02" and hour == 9:
                continue        
            elif day == "02" and hour == 10:
                continue        
            elif day == "02" and hour == 15:
               continue            
            for minute in range(0, 6):
                if day == "02" and hour == 11 and minute < 5:
                    continue            
                elif day == "02" and hour == 14 and minute > 4:
                    continue            
                time.append(get_single_time(prefix, hour, minute))       
    
    
    return time


def df_list_disc(date):
    '''
    Store all the discussion file of a given date and store them as a 
    dataframein a list.
    Input:
      date: string of date, e.g. 2018-03-06
    Return: a list of dataframe
    '''

    df_list = []
    time = get_time_disc(date)
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

    return df_list


def list_to_df(date, df_list, column_names, key_list, freq):
    ''''
    Merge the dataframes in the list in a specific date
    into a total dataframe based on the gap of prediction.
    Inputs:
      date: string of date, e.g. 2018-03-06
      df_list: a list of dataframe
      column_names: list of columns
      key_list: list of columns to merge data frames
      freq: string, gap of prediction(predict price one hour later, half an 
            hour later or ten minutes later), in one of the following string,
            'low'(ten minutes later), 'medium'(half an hour later), 'high'
            (one hour later)
    Return: a dataframe
    '''
    total = pd.DataFrame(columns=column_names)
    for ind, df in enumerate(df_list):
        if freq == 'low':
            cutoff = 3
        elif freq == 'medium':
            cutoff = 5
        elif freq == 'high':
            cutoff = 8

        if ind >= cutoff:
            df_total = df.merge(df_list[ind - 8], on = \
                                key_list).merge(df_list[ind - cutoff  + 1], \
                                on = key_list).merge(df_list[ind - cutoff + \
                                                     2], on = key_list)
            df_total.columns = column_names
            total = pd.concat([total, df_total], axis = 0)

    return total 


# final version only need 'high' for list to df
def get_discuss_df(date):
    '''
    Get total dataframe of discussion from raw files
    Input:
      date: string of date, e.g. 2018-03-06
    Return: a dataframe
    '''

    discussion_list = df_list_disc(date)
    discuss_df = list_to_df(date, discussion_list, COLUMN_DISC, 
                            ['name'], 'high').reset_index().drop(["index"], \
                            axis = 1)

    return discuss_df


def df_list_price(date):
    '''
    Get the dataframes of price during different time within a day and store 
    them in a list. 
    Input:
      date: string of date, e.g. 2018-03-06
    Return: a list
    '''
    
    focus, price = open_files(date)
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
        
    return price_df_list



def get_price_df(date):
    '''
    Transform all the raw price files into a data frame
    Input:
      date: string of date, e.g. 2018-03-06
    Return: a datafame
    '''

    price_df_list = df_list_price(date)
    price_df = list_to_df(date, price_df_list, COLUMN_PRICE, \
                          ['code', 'name'], 'high').reset_index().drop(["index"], axis = 1)
    
    return price_df


def get_total_df(date):
    '''
    Get a total data frame from raw data with both price and discussion 
    information.
    Input:
      date: string of date, e.g. 2018-03-06
    Return: a datafame
    '''

    price_df = get_price_df(date)
    discuss_df = get_discuss_df(date)
    total_df = pd.merge(price_df, discuss_df, on = ['name', \
                                                    'time', 'time_1', \
                                                    'time_2', 'time_3'])
    total_df = total_df[COLUMN_TOTAL]
    
    return total_df


def add_company(date):
    '''
    Add company info to the dataframe created from the price and discussion
    raw data.
    Input:
      date: string of date, e.g. 2018-03-06
    Return: a dataframe
    '''

    total = get_total_df(date)
    total["mkt_cap"] = np.nan
    total["kospi"] = np.nan
    total["kosdaq"] = np.nan
    total["trash"] = np.nan
    
    for index, row in total.iterrows():
        mkt_cap = MKT_CAP[row["code"]]

        kospi = 1 if row["code"] in KOSPI else 0

        kosdaq = 1 if row["code"] in KOSDAQ else 0

        trash = 1 if row["code"] in TRASH else 0

        total.set_value(index,'mkt_cap', mkt_cap)
        total.set_value(index,'kospi', kospi)
        total.set_value(index,'kosdaq', kosdaq)
        total.set_value(index,'trash', trash)  
    
    return total

def transform_df(date):
    '''
    Return a total data frame created from market, price and discussion 
    raw data with all numeriacal variables' values as floats. 
    Input:
      date: string of date, e.g. 2018-03-06
    Return: a dataframe
    '''

    total = add_company(date)
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


def total_date_df(dates):
    '''
    Make a complete dataframe with info from raw data of discussion, 
    price and market index combining specidfic dates defined. 
    Inputs:
      dates: list of dates, e.g. ['2018-02-28', '2018-03-02']
    Return: a dataframe
    '''
    
    total_df = pd.DataFrame(columns=COLUMN_TOTAL)
    for date in dates:
        df = transform_df(date)
        if dates == '2018-02-27':
            df = df[~df['time'].isin(DEL_TIME)]
        total_df = pd.concat([total_df, df])
    total_df = total_df.reset_index().drop(["index"], axis = 1)

    return total_df 


def complete_df(dates):
    '''
    Make a complete dataframe with modified variables ready for analyze.
    Input:
      dates: a list of dates to mark the data
    Return: a dataframe     
    '''
    
    total_df = total_date_df(dates)
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
                              ((total_df["price_2"] - total_df["price_3"]) <\
                               0).astype(int)) - (((total_df["price_1"] - \
                               total_df["price_2"]) > 0).astype(int) + \
                               ((total_df["price_2"] - total_df["price_3"]) >\
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
        r_var = var + "_trend"
        var_3, var_2, var_1 = var + "_3", var + "_2", var + "_1"
        total_df[r_var] = ((((total_df[var_3]) - \
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
        pi_ind = KOSPI_DIFF[row['time']]
        daq_ind = KOSDAQ_DIFF[row['time']]
        total_df.set_value(index,'kospi_ind', pi_ind)
        total_df.set_value(index,'kosdaq_ind', daq_ind)
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

        mkt_change = row['kospi'] * pi_ind + row['kosdaq'] * daq_ind
        total_df.set_value(index,'mkt_change', mkt_change)
        total_df.set_value(index,'early_mor', early_mor)
        total_df.set_value(index,'morning', morning)
        total_df.set_value(index,'lunch', lunch)
        total_df.set_value(index,'afternoon', afternoon)
        total_df.set_value(index,'late', late)
        total_df.set_value(index,'time_slot', num)
        total_df.set_value(index,'ko_inter', pi_ind * daq_ind)
        per_now = 100 * row['price_dif_3']/row['yesterday_closing_price']
        total_df.set_value(index, 'per_now', per_now)
        alpha = per_now - mkt_change
        total_df.set_value(index, 'alpha', alpha)
        
        total_df.set_value(index, 'kospi_1', KOSPI_NOW[row["time_1"]])
        total_df.set_value(index, 'kospi_2', KOSPI_NOW[row["time_2"]])
        total_df.set_value(index, 'kospi_3', KOSPI_NOW[row["time_3"]])
        total_df.set_value(index, 'kospi_answer', KOSPI_NOW[row["time"]])
     
        total_df.set_value(index, 'kosdaq_1', KOSDAQ_NOW[row["time_1"]])
        total_df.set_value(index, 'kosdaq_2', KOSDAQ_NOW[row["time_2"]])
        total_df.set_value(index, 'kosdaq_3', KOSDAQ_NOW[row["time_3"]])
        total_df.set_value(index, 'kosdaq_answer', KOSDAQ_NOW[row["time"]])

    
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

    for var in TO_SQ:
        col_name = var + '_sq'
        total_df[col_name] = np.nan
        for index, row in total_df.iterrows():
            sqr = row[var] ** 2
            total_df.set_value(index, col_name, sqr)


    return total_df

DATES = ['2018-02-14', '2018-02-20', '2018-02-21', '2018-02-22', \
         '2018-02-23', '2018-02-26', '2018-02-27', '2018-02-28', \
         '2018-03-02', '2018-03-05', '2018-03-06', '2018-03-07']

total_df = complete_df(DATES)
total_df.to_json('df_Mar_07.json', orient='values')
