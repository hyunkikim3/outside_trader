import bs4
import urllib3

def scrape_price_dayhigh(code):
	'''
	code: string
    '''

    target = "http://finance.naver.com/item/sise_day.nhn?code=" + code + "&page=1"
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