#run the file and save name of stocks with opening price greater than 
#closing price the day before out of the search ranking focus group into 
#a json file

import pandas as pd
import numpy as np
import json
import re
import bs4
import urllib3

with open("2018-03-06_focus_group.json", "r", \
	      encoding="UTF-8") as focus_group:
    FOCUS = json.load(focus_group)

with open("krx_code.json", "r", encoding="UTF-8") as krx:
    KRX_CODE = json.load(krx)

def opening_increase(code, date):
    
    '''
    Return whether a stock with given code in the specific date has 
    opening price higer than closing price. cannot get info before seven
    days ago.

    Inputs:
      code: string, '035720'
      date: string, e.g. '2018-03-08'

    Return: boolean
    '''

    target = "http://finance.naver.com/item/sise_time.nhn?code=" + code + \
             "&thistime=" + re.sub('[-]', '', date) + "090001&page=1"
    pm = urllib3.PoolManager()
    html = pm.urlopen(url=target, method="GET").data
    soup = bs4.BeautifulSoup(html, 'lxml')
    
    data_list = soup.find_all("tr")[2].find_all("td",class_="num")
    price = data_list[0].text
    
    return price != '\xa0' and "상승" in str(soup)

def get_increase(date):
    '''
    get the list of focus stocks with opening price increased inside a list.
    
    Input: 
      date: string, e.g. '2018-03-08'

    Return: a list  
    '''
    rv = []
    c = 0
    for stock in FOCUS:
        c +=1
        if opening_increase(KRX_CODE[stock], date):
            rv.append(stock)
            print(c)

    return rv


#example for 2018-03-06
filename = "2018-03-06_opening_increase.json"
increase = get_increase('2018-03-06')
with open(filename,"w", encoding='UTF-8') as f:
    	json.dump(increase, f)
    	
