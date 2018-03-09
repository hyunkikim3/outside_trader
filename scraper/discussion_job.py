#run this python file and scrape discussion information of fucus group
#the file take 2018-03-06 as an example

import pandas as pd
import json
from datetime import tzinfo, timedelta, datetime
from pytz import timezone
from apscheduler.schedulers.blocking import BlockingScheduler
import scrape_discussion

with open("krx_code.json", "r", encoding="UTF-8") as krx:
    KRX_CODE = json.load(krx)

with open("2018-03-06_opening_increase.json") as focus_group:
    FOCUS = json.load(focus_group)

def job_function():
    '''
    The job function to scrape discussion info of focus group
    '''
    
    rv = []
    tz = timezone('Asia/Seoul')
    seoul_now = datetime.now(tz)
    
    c = 0
    for stock in FOCUS:
        d = scrape_discussion.scrape_discussion(KRX_CODE[stock])
        d["name"] = stock
        d["time"] = str(seoul_now)[:16]
        rv.append(d)
        c += 1
        print(c)
        print(d)

    #filename = "discussion_" + str(seoul_now)[:16] + ".json"
    with open('y.json',"w", encoding='UTF-8') as f:
        json.dump(rv, f)
    
    return 

sched = BlockingScheduler()
sched.add_job(job_function, 'cron', month='3', day='7-28', hour='0-23', \
              minute='0-59/10')
sched.start()



