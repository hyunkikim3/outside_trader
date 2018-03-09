#run this file then starts real time scraping every ten minutes in March

import bs4
import urllib3
from datetime import datetime, timezone
import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
import json

TARGET = "http://finance.naver.com/sise/lastsearch2.nhn"

def find_all_stock():
    '''
    scrape the real time search ranking table and store the ranking
    and stock name in a dictionary.

    Input

    Return
    ''' 

    pm = urllib3.PoolManager()
    html = pm.urlopen(url=TARGET, method="GET").data
    soup = bs4.BeautifulSoup(html, 'lxml')
    div_tag = soup.find_all("div", class_ ="box_type_l")[0]
    raw_stock_list = div_tag.find_all("tr")[2:47]
    stock_dict = {}
    for tag in raw_stock_list:
        if raw_stock_list.index(tag) % 8 not in [5, 6, 7]:
            rank = tag.find_all("td", class_ = "no")[0].text
            name = tag.find_all("td", class_ = \
                   "no")[0].next_sibling.next_sibling.find_all("a")[0].text
            stock_dict[rank] = name

    return stock_dict

def job_function():
    '''
    The job function of the cron job to scrape real time scraper.
    '''
    
    rank = find_all_stock()
    tz = pytz.timezone('Asia/Seoul')
    seoul_now = datetime.now(tz)
    
    rank["time"] = str(seoul_now)[:16]
    
    json = json.dumps(rank, ensure_ascii=False)
    f = open(str(rank["time"])+".json","w", encoding='UTF-8')
    f.write(json)
    f.close()
    
    return 


sched = BlockingScheduler()
sched.add_job(job_function, 'cron', month='3', day='1-31', hour='0-23', \
              minute='0-59/10')
sched.start()