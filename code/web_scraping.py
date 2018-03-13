import numpy as np
import pandas as pd
import re
import json
import bs4
import urllib3
from datetime import datetime, tzinfo, timedelta
from pytz import timezone
from apscheduler.schedulers.blocking import BlockingScheduler

try:
    with open("../data/krx_code.json", "r", encoding="UTF-8") as krx:
        KRX_CODE = json.load(krx)

except FileNotFoundError as e:
    print(e)


def save_krx_code():
    '''
    Direct copy from http://excelsior-cjh.tistory.com/entry/5-Pandas를-이용한-Naver금융에서-주식데이터-가져오기
    
    save the table mapping different stock names and stock codes into file
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
    '''
    save the dictionary mappping different stock codes to company 
    names to json file
    
    Return: a list
    '''
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
    
    Input:
      save: boolean, default = False, if true save the ranking 
             dictionary into a file
    
    Return: a dictionary
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
    scrape the the maximum and minimum of a stock in the previous 
    five trading days and save them as a tuple.
    
    Input: 
      code: string of stock, e.g. '035720'
      
    Return: a tuple
    '''
    target = "http://finance.naver.com/item/sise_day.nhn?code=" + code + \
             "&page=1"
    pm = urllib3.PoolManager()
    html = pm.urlopen(url=target, method="GET").data
    soup = bs4.BeautifulSoup(html, 'lxml')
    data_list = soup.find_all("tr")[2:7]
    price = []
    for data in data_list:
        new_list = data.find_all("td",class_="num")
        high = new_list[3].text.split(",")
        high = "".join(high)
        low = new_list[4].text.split(",")
        low = "".join(low)
        price.append(high)
        price.append(low)
    
    return max(price), min(price)


def filter_quiet_stock(date, save=False):
    '''
    Filter and only save the stocks in the ranking table file 
    of the given date, which appear only once in ranking overnight 
    price volatility are less than 50 percent in five days. Return
    them in a list. 
    
    Input: 
      date: string of date, e.g. "2018-03-07", korea work days only
      save: boolean, True if save the ranking table in a new file
    
    Return: a list
    '''
    time = []

    day_before = datetime(int(date[:4]), int(date[5:7]), \
                int(date[8:])) - timedelta(days=1)
    yesterday = "-".join([str(day_before.year), \
                (str(day_before.month) if day_before.month > 10 \
                else "0" + str(day_before.month)),\
                (str(day_before.day) if day_before.day > 10 \
                else "0" + str(day_before.day))])
    
    for hour in range(20,24):
        for minute in range(0, 6):
            time.append("../data/ranking/" + yesterday + "/" + yesterday + "-" + \
                        (("0" + str(hour)) if hour <= 9 \
                        else str(hour)) + "-" + str(minute) + "0.json")

    for hour in range(0,7):
        for minute in range(0, 6):
            if hour == 0 and minute == 1:
                continue
            time.append("../data/ranking/" + date + "/" + date + "-" + \
                        (("0" + str(hour)) if hour <= 9 \
                        else str(hour))+ "-" + str(minute) + "0.json")

    counter = {}

    for t in time:
        try:
            with open(t) as ranking_table:
                rank = json.load(ranking_table)

                for v in rank.values():
                    if v == t[11:21] + " " + t[22:24] + ":" + t[25:27]:
                        pass
                    elif counter.get(v) == None:
                        counter[v] = 1
                    else:
                        counter[v] = counter.get(v) + 1
        except FileNotFoundError as e:
            print(e)

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
    Scrape the real time discussion forum information (post_num, unique_id, 
    click, like, dislike) of stock of given stock code from the day before 
    till the moment(current time 2018-2-14 15:00 then scrape cumulative info 
    from 2018-2-13 00:00 to now) and save it inside a dictionary.
    
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


def save_discussion(date):
    '''
    Return the disussion info of the stocks in the focus group filtered in 
    the search ranking table and store them in a list. Then save the list
    in another json file. 
    
    Input:
      date: string of date,  e.g. '035720'
      
    Return: a list
    '''
    
    try:
        with open("../data/discussion/" + date + "_focus/" + date \
                  + "_focus_group.json", "r", encoding="UTF-8") as focus:
            focus_group = json.load(focus)
    
    except FileNotFoundError as e:
        print(e)
        return None

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
    Return a list of stock names of stocks in the focus group whose 
    opening price on the date given is higer than closing price the 
    date before. 
    
    p.s. The info cannot be got after seven days of the data, for the
    the data would be deleted online
    days ago.
    
    Inputs:
      date: string of date, e.g. '2018-03-08', korea work days only
      save: boolean, default is False, if True, save the list in another 
            file in the save directory of the focus group info. 
            
    Return: a list
    '''
    
    try:
        with open("../data/discussion/" + date + "_focus/" + date \
                  + "_focus_group.json", "r", encoding="UTF-8") as focus:
            focus_group = json.load(focus)
    
    except FileNotFoundError as e:
        print(e)
        return None

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
    Scrape the history price information of the stock with the given code 
    on the given date, given time, and store the information in a dictionaty, 
    including price, price difference compared with one minute ago, selling 
    price, buying price, volume of trade and variation of price. It cannot 
    scrape information more than seven days before, and workday only.
    
    Inputs: 
      code: the string of stock code, e.g. '035720'
      time: the string of date, e.g. '201802280901'
      
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
    Get market price index of the given date, both kospi index  and 
    kosdaq index, and store them in a tuple.
    
    Input:
      date: string of date, e.g. '2018-03-08', korea workdays only
      save: boolean, if True save the index info on the given date 
            inside a json file
            
    Return: a tuple
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
            
            try:
                index = float(tag.find("td",class_ = "number_1"\
                                      ).string.replace(",", ""))
            
            except ValueError as e:
                print(e)
                return None

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
                    kospi[date + " last_closing"] = index - delta
                kospi[date + " " + t[0:2] + ":" + t[2:4]] = index

            else:
                if flag:
                    kosdaq[date + " last_closing"] = index - delta

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


def save_price(date):
    '''
    Store the price info of every focus stock on the given date in a list
    and store all the lists in a list.
    
    It takes more than thrity minutes to finish running. 
    
    Input: 
      date: string of date, e.g. '2018-03-08', korea work days only
    
    Return: list of lists 
    '''
    opening_increase = "../data/price/" + date + "_price/" + date +\
                            "_opening_increase.json"
    
    try:
        with open(opening_increase, 'r', encoding='UTF-8') as f:
            increased = json.load(f)
    
    except FileNotFoundError as e:
        print(e)
        return None
    
    time = []
    for hour in range(9,16):
        for minute in range(0, 6):
            if hour != 15 or minute < 4:
                time.append("../data/discussion/" + date + \
                "_focus/discussion_" + date + "-" + (("0" + str(hour)) \
                if hour <= 9 else str(hour))+ "-" + str(minute) + "0.json")
    
    df = pd.DataFrame(columns=["code", "name", "time", "price", "price_dif",\
                                        "sell", "buy", "volume", "variation"])
    
    for t in time:
        with open(t, 'r', encoding='UTF-8') as f:
            discussion = json.load(f)
        for d in discussion:
            if d["name"] in increased:
                row = pd.DataFrame(columns=["code", "name", "time", "price", "price_dif",\
                                            "sell", "buy", "volume", "variation"],\
                               data = [[KRX_CODE[d["name"]], d["name"], d["time"], np.nan, np.nan,\
                                       np.nan, np.nan, np.nan, np.nan]])
                df = df.append(row)
    
    df = df.reset_index()
    
    for idx, row in df.iterrows():
        timestamp = row["time"]
        t = re.sub('[ :-]', '', timestamp)

        d = scrape_price_history(row["code"], t)

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
    
    with open(date + "_price.json","w", encoding='UTF-8') as f:
        json.dump(rv, f, ensure_ascii=False)

    return rv


def ranking_dictionary(target):
    '''
    srape the real time search ranking of naver finance website and store them
    in a dictionary
    
    Input:
      target: string of html, "http://finance.naver.com/sise/lastsearch2.nhn"
    
    Return: a dictionary
    '''
    
    pm = urllib3.PoolManager()
    html = pm.urlopen(url=target, method="GET").data
    soup = bs4.BeautifulSoup(html, 'lxml')
    div_tag = soup.find_all("div", class_ ="box_type_l")[0]
    raw_stock_list = div_tag.find_all("tr")[2:47]
    stock_dict = {}
    for tag in raw_stock_list:
        if raw_stock_list.index(tag) % 8 not in [5, 6, 7]:
            rank = tag.find_all("td", class_ = "no")[0].text
            name = tag.find_all("td", class_ = "no")[0].next_sibling.next_sibling.find_all("a")[0].text
            stock_dict[rank] = name
            
    return stock_dict


def ranking_job_function():
    '''
    save the real tme search ranking data in a json file
    '''
    import json
    
    rank = ranking_dictionary("http://finance.naver.com/sise/lastsearch2.nhn")
    tz = pytz.timezone('Asia/Seoul')
    seoul_now = datetime.now(tz)
    
    rank["time"] = str(seoul_now)[:16]
    
    json = json.dumps(rank, ensure_ascii=False)
    f = open(str(rank["time"])+".json","w")
    f.write(json)
    f.close()
    
    return None


def discussion_job_function():
    '''
    The job function to scrape discussion info of focus group
    '''
    import json
    rv = []
    tz = timezone('Asia/Seoul')
    seoul_now = datetime.now(tz)
    
    for stock in FOCUS:
        d = scrape_discussion.scrape_discussion(KRX_CODE[stock])
        d["name"] = stock
        d["time"] = str(seoul_now)[:16]
        rv.append(d)
    
    filename = "discussion_" + str(seoul_now)[:16] + ".json"
    with open('y.json',"w", encoding='UTF-8') as f:
        json.dump(rv, f)
    
    return None


def cron(date):
    '''
    
    '''
    try:
        with open("../data/krx_code.json", "r", encoding="UTF-8") as f:
            krx = json.load(f)
    except FileNotFoundError as e:
        print(e)
        return None
    
    try:
        with open("../data/price/" + date + "_price/" +\
                  date + "_opening_increase.json") as f:
            focus = json.load(f)
    except FileNotFoundError as e:
        print(e)
        return None

    sched = BlockingScheduler()
    sched.add_job(ranking_job_function, 'cron', month='1-12', day='1-31', hour='0-23', \
              minute='0-59/10')
    sched.add_job(discussion_job_function, 'cron', month='1-12', day='1-31', hour='0-23', \
                  minute='0-59/10')
    sched.start()
    
    return None
