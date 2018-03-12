#run this file and get market index (KOSPI and KOSDAQ) for the specific date


import json
import re
import bs4
import urllib3

#take the date 2018-03-06 as an example 
DATE_LIST = ["2018-03-06"]
MKT_LIST = ["KOSPI", "KOSDAQ"]

def get_time():
    '''
    Get a list of time from 9 am to 3:30 pm
    '''
    time = []
    for i in range(9,16):
        i = str("0") + str(i) if i < 10 else str(i)
        for j in range(0, 60, 10):
            j = str("00") if j == 0 else str(j)
            time.append(i+j)
    del time[41]
    del time[40]
    time[0] = "0901"

    return time

def get_market_price(diction, mkt, date, minute):
    '''
    Get market price index for
    '''
    
    target = "finance.naver.com/sise/sise_index_time.nhn?code=" + mkt + \
             "&thistime=" + re.sub('[-]', '', date) + minute + "01&page=1"
    pm = urllib3.PoolManager()
    html = pm.urlopen(url=target, method="GET").data
    soup = bs4.BeautifulSoup(html, 'lxml')

    tag = soup.find_all("tr")[2]
    
    index = float(tag.find("td", class_ = "number_1").string.replace(",", \
    	          ""))

    if "상승" in str(tag):
        direction = 1
    elif "하락" in str(tag):
        direction = -1
    else:
        direction = "error"

    magnitude = tag.find("td", class_ = "rate_down").text.strip("\t\n")
    
    delta = direction * float(magnitude)
    
    diction[date + " " + minute[0:2] + ":" + minute[2:4]] = ((index * 100) \
    	    / (index - delta)) - 100
    
    return 


for mkt in MKT_LIST:
    for date in DATE_LIST:
        rv = {}
        c = 0
        for time in get_time():
            c += 1
            get_market_price(rv, mkt, date, time)
            print(c)
        filename = mkt + "_" + date + ".json"
        with open(filename,"w", encoding='UTF-8') as f:
            json.dump(rv, f)
