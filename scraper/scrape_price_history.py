import bs4
import urllib

def scrape_price_history(code, time):
	'''
	code: string
	time: string, e.g. 201802090910
	'''

    target = "http://finance.naver.com/item/sise_time.nhn?code=" + code + "&thistime=" + time + "01&page=1"
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