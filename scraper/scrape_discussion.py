import bs4
import urllib3
from datetime import tzinfo, timedelta, datetime
from pytz import timezone

def scrape_discussion(code):
    page_num = 1
    post_dic = {}
    post_num = 0
    click = 0
    like = 0
    dislike = 0
    unique_id = set()
    while page_num <= 10: ###
        target = "http://finance.naver.com/item/board.nhn?code=" + code + "&page=" + str(page_num)
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
                pre_time = datetime(int(date_list[0]), int(date_list[1]), int(date_list[2]))
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