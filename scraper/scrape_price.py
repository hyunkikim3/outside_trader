import bs4
import urllib3
import csv

def scrape_price(code):
    '''
    scrape real time price info of a stock with given stock code (yester day 
    closing price, today opening price, today highest price, today possible 
    highest, today possible lowest, trade volume and trade amount) and save it
    into a dictionary

    Input: 
      code: the string of stock code, e.g. 035720

    Return: dictionary
    '''

    target = "http://finance.naver.com/item/coinfo.nhn?code=" + code + "#"
    pm = urllib3.PoolManager()
    html = pm.urlopen(url=target, method="GET").data
    soup = bs4.BeautifulSoup(html, 'lxml')
    data_list = soup.find_all("dl", class_ = "blind")[0].find_all("dd")
    data_dic = {}
    first_line = data_list[3].text.split()
    data_dic["price"] = first_line[1]
    if first_line[5] == "마이너스":
        data_dic["value_dif"] = "-" + first_line[4]
        data_dic["value_dif_per"] = "-" + first_line[6]
    else:
        data_dic["value_dif"] = first_line[4]
        data_dic["value_dif_per"] = first_line[6]
    var_list = ["yestday_close", "today_open", "today_high", "highest_poss", \
                "today_low", "lowest_poss", "trade_vol", "trade_amount"]
    num = 4
    for var in var_list:
        data_dic[var] = data_list[num].text.split()[1]
        num += 1
    
    return data_dic

def add_price_to_file(filename, code):

	'''
	save real time price info of a stock with given stock code and save it
	to csv file. 

	Input: 
	  filename: string of csv filename
	  code: string of stock code

	Return
	'''

	fieldnames = ["price", "value_dif", "value_dif_per", "yestday_close", \
					"today_open", "today_high", "highest_poss", "today_low", \
					"lowest_poss", "trade_vol", "trade_amount"]
	row = scrape_price(code)
	with open(filename, "w", encoding='UTF-8') as f:
		writer = csv.DictWriter(f, fieldnames = fieldnames)
		writer.writeheader()
		writer.writerow(row)
        
	return    