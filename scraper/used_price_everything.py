#run this file to get price info of one day into a json 
#file take 2018-03-06 as an example

import pandas as pd
import numpy as np
import json
import re
import scrape_price_hour_min

with open("krx_code.json", 'r', encoding='UTF-8') as krx:
    KRX_CODE = json.load(krx)

with open("2018-03-06_opening_increase.json", 'r', encoding='UTF-8') as oi:
    INCREASE = json.load(oi)

PREFIX = "discussion_2018-03-06"

def get_discussion_time(prefix):
    '''
    Get all the discussion time filename of a singular day and save them 
    in a list.

    Input:
      prefix: string of discussion file prefix, e.g. "discussion_2018-03-06"

    Return: a list
    '''

    time = []
    for hour in range(9,16):
        for minute in range(0, 6):
            time.append(prefix + " " + (("0" + str(hour)) if hour <= 9 else \
                        str(hour))+ ":" + str(minute) + "0.json")

    del time[41]
    del time[40]

    return time

def combine_price(prefix):
    '''
    Combine price and discussion info into a dataframe with on the given date
    and save them in a list

    Input:
      prefix: string of discussion file prefix, e.g. "discussion_2018-03-06"

    Return: a list
    '''

    df = pd.DataFrame(columns=["code", "name", "time", "price", \
    	                       "price_dif", "sell", "buy", "volume", \
    	                       "variation"])
    rv = []
    rv.append(df.columns.tolist())
    time = get_discussion_time(PREFIX)
    for t in time:
        with open(t, 'r', encoding='UTF-8') as f:
            discussion = json.load(f)
        for d in discussion:
            if d["name"] in INCREASE:
                row = pd.DataFrame(columns=["code", "name", "time", \
                                            "price", "price_dif", "sell", \
                                            "buy", "volume", "variation"], \
                                   data = [[KRX_CODE[d["name"]], d["name"], \
                                            d["time"], np.nan, np.nan,\
                                            np.nan, np.nan, np.nan, np.nan]])
                df = df.append(row)

    df = df.reset_index()
    for idx, row in df.iterrows():
        timestamp = row["time"]
        t = re.sub('[ :-]', '', timestamp)    
        d = scrape_price_hour_min.scrape_price_history(row["code"], t)    
        df["price"].iloc[idx] = d["price"]
        df["price_dif"].iloc[idx] = d["price_dif"]
        df["sell"].iloc[idx] = d["sell"]
        df["buy"].iloc[idx] = d["buy"]
        df["volume"].iloc[idx] = d["volume"]
        df["variation"].iloc[idx] = d["variation"]
        rv.append(row.tolist())
    

    return rv

#take 2018-03-06 as an example
rv = combine_price(PREFIX)
filename = "2018-03-06_price_everything.json"
with open(filename,"w", encoding='UTF-8') as f:
        json.dump(rv, f)


