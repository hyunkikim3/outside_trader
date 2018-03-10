import numpy as np
import pandas as pd
import re
import json
import bs4
import urllib3
from datetime import datetime, tzinfo, timedelta
from pytz import timezone

with open("krx_code.json", "r", encoding="UTF-8") as krx:
    KRX_CODE = json.load(krx)

def save_krx_code():
    '''
    Direct copy
    '''
    code_df = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?\
                            method=download&searchType=13', header=0)[0]
    code_df.종목코드 = code_df.종목코드.map('{:06d}'.format)
    code_df = code_df[['회사명', '종목코드']]
    code_df = code_df.rename(columns={'회사명': 'name', '종목코드': 'code'})

    rv = {}
    for idx, row in code_df.iterrows():
        rv[row["name"]] = row["code"]

    with open("krx_code.json","w", encoding='UTF-8') as f:
        json.dump(rv, f, ensure_ascii=False)

    return rv


def scrape_company_info(code):
    '''
    Scrape company size info of the stocks with given code and store the
    info in a dictionary
    Input: 
      code: string of stock code, e.g. '035720'
    Retun: a dictionary
    '''

    target = "http://finance.naver.com/item/coinfo.nhn?code=" + code + "#"
    pm = urllib3.PoolManager()
    html = pm.urlopen(url=target, method="GET").data
    soup = bs4.BeautifulSoup(html, 'lxml')
    tag_1 = soup.find_all('tr', class_ = "strong")[0]
    size = tag_1.find("em").text.strip("\t\n") + tag_1.find("em").next_sibling
    noise = '\n\t'
    tag_2 = tag_1.next_sibling.next_sibling
    market = tag_2.find("td").text
    for letter in noise:
        size = size.replace(letter, '')
        market = market.replace(letter, '')
    r_dic = {}
    r_dic["code"] = code
    r_dic["size"] = size
    r_dic["market"] = market
    
    return r_dic


def save_company_info():
    
    company_df = pd.DataFrame(list(KRX_CODE.items()), 
                 columns=['name', 'code'])

    with open("company_info.json", "w",  encoding='UTF-8') as f:
        rv = []
        for index_df, row_df in company_df.iterrows():
            r_dic = scrape_company_info(row_df["code"])
            r_dic["name"] = row_df["name"]
            data = [r_dic["name"], r_dic["code"], \
                    r_dic["market"], r_dic["size"]]
            rv.append(data)

        json.dump(rv, f, ensure_ascii=False)

    return rv


def scrape_ranking(save=False):
    '''
    scrape the real time search ranking table and store the ranking
    and stock name in a dictionary.
    Input
    Return
    '''
    pm = urllib3.PoolManager()
    target = "http://finance.naver.com/sise/lastsearch2.nhn"
    html = pm.urlopen(url=target, method="GET").data
    soup = bs4.BeautifulSoup(html, 'lxml')
    div_tag = soup.find_all("div", class_ ="box_type_l")[0]
    raw_stock_list = div_tag.find_all("tr")[2:47]
    rv = {}
    for tag in raw_stock_list:
        if raw_stock_list.index(tag) % 8 not in [5, 6, 7]:
            rank = tag.find_all("td", class_ = "no")[0].text
            name = tag.find_all("td", class_ = \
                   "no")[0].next_sibling.next_sibling.find_all("a")[0].text
            rv[rank] = name

    if save:
        tz = timezone('Asia/Seoul')
        
        seoul_now = datetime.now(tz)

        rv["time"] = str(seoul_now)[:16]

        filename = str(seoul_now)[:10] + "-" + str(seoul_now)[11:13] +\
                   "-" + str(seoul_now)[14:16] + ".json"


        with open(filename,"w", encoding='UTF-8') as f:
            json.dump(rv, f, ensure_ascii=False)

    return rv


def scrape_5day_high_low(code):
    '''
    scrape the tuple of the maximum and minimum of a stock today
    and save them as a tuple.
    Input: 
      code: string of stock
    Return: a tuple
    '''

    target = "http://finance.naver.com/item/sise_day.nhn?code=" + code + \
             "&page=1"
    pm = urllib3.PoolManager()
    html = pm.urlopen(url=target, method="GET").data
    soup = bs4.BeautifulSoup(html, 'lxml')
    data_list = soup.find_all("tr")[2:7]
    high_list = []
    low_list = []
    for data in data_list:
        new_list = data.find_all("td",class_="num")
        high = new_list[3].text.split(",")
        high = "".join(high)
        low = new_list[4].text.split(",")
        low = "".join(low)
        high_list.append(high)
        low_list.append(low)
    
    return max(high_list), min(low_list)


def filter_focus_group(date, save=False):
    '''
    input = "2018-02-20"

    '''
    time = []

    day_before = datetime(int(date[:4]), int(date[5:7]), \
                int(date[8:])) - timedelta(days=1)
    yesterday = "-".join([str(day_before.year), \
                (str(day_before.month) if day_before.month > 10 \
                else "0" + str(day_before.month)), str(day_before.day)])

    for hour in range(20,24):
        for minute in range(0, 6):
            time.append(yesterday + "/" + yesterday + "-" + \
                        (("0" + str(hour)) if hour <= 9 \
                        else str(hour)) + "-" + str(minute) + "0.json")

    for hour in range(0,7):
        for minute in range(0, 6):
            time.append(date + "/" + date + "-" + \
                        (("0" + str(hour)) if hour <= 9 \
                        else str(hour))+ "-" + str(minute) + "0.json")

    counter = {}

    for t in time:
        with open(t) as table:
            rank = json.load(table)
            
            for v in rank.values():
                if v == t[11:21] + " " +t[22:24] + ":" + t[25:27]:
                    pass
                elif counter.get(v) == None:
                    counter[v] = 1
                else:
                    counter[v] = counter.get(v) + 1

    rv = []

    for k, v in counter.items():
        if KRX_CODE.get(k) != None and v == 1:

            high, low = scrape_5day_high_low(KRX_CODE[k])

            if high != '\xa0' and low != '\xa0' and int(low) != 0:
                if float(high) / float(low) < 1.15:
                    rv.append(k)

    if save:

        with open(date + "_focus_group.json","w", encoding='UTF-8') as f:
            json.dump(rv, f, ensure_ascii=False)

    return rv


def scrape_discussion(code):
    '''
    scrape the real time discussion forum information (post_num, unique_id, \
    click, like, dislike) of a stock with given stock code from the day \
    before (current time 2018-2-14 15:00 then scrape cumulative info from \
    2018-2-13 00:00 to now) and save it inside a dictionary.
    Input:
      code: string of stock code, e.g. '035720'
    Return: a dictionary
    '''

    page_num = 1
    post_dic = {}
    post_num = 0
    click = 0
    like = 0
    dislike = 0
    unique_id = set()
    while page_num < 100:
        target = "http://finance.naver.com/item/board.nhn?code=" + code + \
                 "&page=" + str(page_num)
        pm = urllib3.PoolManager()
        html = pm.urlopen(url=target, method="GET").data
        soup = bs4.BeautifulSoup(html, 'lxml')
        div_tag = soup.find_all("div", class_ ="section inner_sub")
        div_row = div_tag[0].find_all("tr")
        for row in div_row[2:25]:
            if div_row.index(row) not in [7, 13, 19]:
                info_list = row.find_all('td')
                time = info_list[0].text
                date, time = time.split(" ")
                date_list = date.split(".")
                time_list = time.split(":")
                pre_time = datetime(int(date_list[0]), int(date_list[1]), \
                           int(date_list[2]))
                korea_time = timezone('Asia/Seoul')
                pre_time = korea_time.localize(pre_time, is_dst = True)
                now = datetime.now(tz = timezone('Asia/Seoul'))
                if now - pre_time <= timedelta(days = 1):
                    post_num += 1
                    unique_id.add(info_list[3].text.strip("\n\t"))
                    click += int(info_list[4].text)
                    like += int(info_list[5].text)
                    dislike += int(info_list[6].text)
                else:
                    post_dic["post_num"] = post_num
                    post_dic["unique_id"] = len(unique_id)
                    post_dic["click"] = click
                    post_dic["like"] = like
                    post_dic["dislike"] = dislike

                    return post_dic

        page_num += 1
    return post_dic


def save_discussion(focus_group):

    rv = []

    tz = timezone('Asia/Seoul')
    seoul_now = datetime.now(tz)

    for stock in focus_group:
        disc_dic = scrape_discussion(KRX_CODE[stock])
        disc_dic["name"] = stock
        disc_dic["time"] = str(seoul_now)[:16]
        rv.append(disc_dic)

    with open("discussion_" + str(seoul_now)[:16] + \
              ".json","w", encoding='UTF-8') as f:
            json.dump(rv, f, ensure_ascii=False)

    return rv


def filter_opening_increase(date, save=False):
    
    '''
    Return whether a stock with given code in the specific date has 
    opening price higer than closing price. cannot get info before seven
    days ago.
    Inputs:
      date: string, e.g. '2018-03-08'
    Return: boolean
    '''

    with open(date + "/" + date + "_focus_group.json", "r", \
          encoding="UTF-8") as focus:
        focus_group = json.load(focus)

    rv = []
    for stock in focus_group:

        target = "http://finance.naver.com/item/sise_time.nhn?code=" + \
                 KRX_CODE[stock] + "&thistime=" + \
                 re.sub('[-]', '', date) + "090001&page=1"
        pm = urllib3.PoolManager()
        html = pm.urlopen(url=target, method="GET").data
        soup = bs4.BeautifulSoup(html, 'lxml')
        
        data_list = soup.find_all("tr")[2].find_all("td",class_="num")
        price = data_list[0].text
        
        if price != '\xa0' and "상승" in str(soup):
            rv.append(stock)

    if save:

        with open(date + "_opening_increase.json","w", encoding='UTF-8') as f:
            json.dump(rv, f, ensure_ascii=False)

    return rv


def scrape_price_history(code, time):
    '''
    Scrape the history price information and store it inside a dictionary
    of the given stock in the given date and time(hour and minute. It cannot 
    scrape information more than ten days before, as long as it's a weekday.
    Input: 
      code: the string of stock code
      time: the string of date 201802280901
    Return: a dictionary
    '''

    target = "http://finance.naver.com/item/sise_time.nhn?code=" + code + \
             "&thistime=" + time + "01&page=1"
    pm = urllib3.PoolManager()
    html = pm.urlopen(url=target, method="GET").data
    soup = bs4.BeautifulSoup(html, 'lxml')
    data_list = soup.find_all("tr")[2].find_all("td",class_="num")
    data_dic = {}
    data_dic["price"] = data_list[0].text
    image = data_list[1].find("img")
    if image != None:
        if image["alt"] == "상승":
            data_dic["price_dif"] = data_list[1].text.strip("\n\t")
        else:
            data_dic["price_dif"] = "-" + data_list[1].text.strip("\n\t")
    else:
        data_dic["price_dif"] = 0
    data_dic["sell"] = data_list[2].text
    data_dic["buy"] = data_list[3].text
    data_dic["volume"] = data_list[4].text
    data_dic["variation"] = data_list[5].text
    
    return data_dic


def scrape_market_history(date, save=False):
    '''
    Get market price index for
    '''

    time = []
    for i in range(9,16):
        i = str("0") + str(i) if i < 10 else str(i)
        for j in range(0, 60, 10):

            if i == "09" and j == 0:
                time.append("0901")

            elif int(i) != 15 or j < 40:
                j = str("00") if j == 0 else str(j)
                time.append(i+j)

    market = ("KOSPI", "KOSDAQ")
    kospi = {}
    kosdaq = {}

    for mkt in market:

        flag = True

        for t in time:

            target = "finance.naver.com/sise/sise_index_time.nhn?code=" + mkt + \
                     "&thistime=" + re.sub('[-]', '', date) + t + "01&page=1"
            pm = urllib3.PoolManager()
            html = pm.urlopen(url=target, method="GET").data
            soup = bs4.BeautifulSoup(html, 'lxml')

            tag = soup.find_all("tr")[2]
            
            index = float(tag.find("td",\
                    class_ = "number_1").string.replace(",", ""))

            if flag:

                if "상승" in str(tag):
                    direction = 1
                elif "하락" in str(tag):
                    direction = -1
                else:
                    direction = 0

                magnitude = tag.find("td", class_ = "rate_down").text.strip("\t\n")
            
                delta = direction * float(magnitude)

            if mkt == "KOSPI":
                if flag:
                    kospi["last_closing"] = index - delta
                kospi[date + " " + t[0:2] + ":" + t[2:4]] = index

            else:
                if flag:
                    kosdaq["last_closing"] = index - delta

                kosdaq[date + " " + t[0:2] + ":" + t[2:4]] = index

            flag = False
    
    if save:
        for mkt in market:
            filename = mkt + "_" + date + ".json"
            if mkt == "KOSPI":
                with open(filename,"w", encoding='UTF-8') as f:
                    json.dump(kospi, f, ensure_ascii=False)
            
            else:
                with open(filename,"w", encoding='UTF-8') as f:
                    json.dump(kosdaq, f, ensure_ascii=False)

    return (kospi, kosdaq)
