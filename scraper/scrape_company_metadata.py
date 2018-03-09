import bs4
import urllib3
import csv

def scrape_company_metadata(code):
	'''
	scrape company info of a stock with given code 
	and save it inside a dictionary.

	Input:
	  code: the string of stock code, e.g. 035720

	Return: a dictionary
	'''

	target = "http://finance.naver.com/item/main.nhn?code=" + code
	pm = urllib3.PoolManager()
	html = pm.urlopen(url=target, method="GET").data
	soup = bs4.BeautifulSoup(html, 'lxml')
	# columns = []
	rows = []
	values = []
	dataset = {}

	all_th = soup.find_all("div", class_ = 
							"sub_section")[4].find_all("th")[3:]
	all_td = soup.find_all("div", class_ = "sub_section")[4].find_all("td")

	for th in all_th:
	# if th["scope"]=="col":
	#     columns.append(th.text.split())
		if th["scope"] == "row":
			rows.append(th.get_text())

	values = [td.text.strip("\r\n\t") for td in all_td]

	# for td in all_td:
	#     values.append(td.text.strip("\r\n\t"))

	for i in range(len(rows)):
		dataset[rows[i]] = values[0+10*(i):10*(i+1)]

	return dataset


def company_file(filename, code):
	'''
	save company data with given stock code in the file with given filename

	Inputs: 
	filename: the string of filename
	code: the string of stock code, e.g. 035720

	Return
	'''

	fieldnames = ['매출액', '영업이익', '당기순이익', '영업이익률', \
				'순이익률', 'ROE(지배주주)', '부채비율', '당좌비율', \
				'유보율', 'EPS(원)', 'BPS(원)','주당배당금(원)', '시가배당률(%)','배당성향(%)']

	row = scrape_company_metadata(code)
	with open(filename, "w", encoding='UTF-8') as f:
		writer = csv.DictWriter(f, fieldnames = fieldnames)
		writer.writeheader()
		writer.writerow(row)

	return