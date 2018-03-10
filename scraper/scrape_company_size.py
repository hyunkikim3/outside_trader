#run this file and save company size info for every company
#in a json file
import bs4
import urllib3
import csv
import pandas as pd
import json
import numpy as np

with open("krx_code.json", 'r', encoding='UTF-8') as krx:
    KRX_CODE = json.load(krx)

def scrape_company_size(code):
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

CODE_DF = pd.DataFrame(list(KRX_CODE.items()), columns=['name', 'code'])

#it takes about half an hour to finish running
with open("company_size.json", "w",  encoding='UTF-8') as f:
    all_data = []
    for index_df, row_df in CODE_DF.iterrows():
        r_dic = scrape_company_size(row_df["code"])
        r_dic["name"] = row_df["name"]
        data = [r_dic["name"], r_dic["code"], r_dic["market"], r_dic["size"]]
        all_data.append(data)

    json.dump(all_data, f)

