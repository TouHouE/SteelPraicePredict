import yfinance as yf
import pandas as pd
import pandas
from datetime import date
from bs4 import BeautifulSoup
from random import randint
import requests as request
'''
    
'''


def wrap_investing(curr_id: int, name: str, start: str, end: str, frequency='Daily') -> pandas.DataFrame:
    post_url = 'https://www.investing.com/instruments/HistoricalDataAjax'
    date_ = date.fromisoformat(start)
    start_date = f'{date_.month}/{date_.day}/{date_.year}'
    date_ = date.fromisoformat(end)
    end_date = f'{date_.month}/{date_.day}/{date_.year}'

    data = {
        'curr_id': str(curr_id),
        'smlID': str(randint(1000000, 99999999)),
        'header': name,
        'st_date': start_date,
        'end_date': end_date,
        'interval_sec': frequency,
        'sort_col': 'date',
        'sort_ord': 'DESC',
    }
    header = {
        'authority': 'www.investing.com',
        'method': 'POST',
        'path': '/instruments/HistoricalDataAjax',
        'accept': 'text/plain, */*; q=0.01',
        'content-type': 'application/x-www-form-urlencoded',
        'x-requested-with': 'XMLHttpRequest',
        'User-Agent': 'Mozilla/5.0'
    }

    req = request.post(url=post_url, data=data, headers=header)


    srrc1 = BeautifulSoup(req.text, 'html.parser')
    srrc1_thead = srrc1.find('thead')
    srrc1_tbody = srrc1.find('tbody')
    # srrc1_tbody.attrs()

    all_label = srrc1_thead.find_all('th')
    tag_dict = {}
    tags = []

    for label in all_label:
        tag_dict[label.text] = []
        tags.append(label.text)

    all_tr = srrc1_tbody.find_all('tr')
    length = len(tags)

    for tr in all_tr:
        data_line = tr.find_all('td')

        for i in range(length):
            if i == 0:
                time = data_line[i]['data-real-value']
                time = date.fromtimestamp(int(time))
                time = f'{time.year}-{time.month}-{time.day}'

                tag_dict[tags[i]].append(time)
            else:
                tag_dict[tags[i]].append(data_line[i].text)
    print(tag_dict)
    df = pd.DataFrame(tag_dict)
    return df


def wrap_yahoo(symbol: str, start: str, end: str, frequency='1d'):
    return yf.download(symbol, start, end, interval=frequency)
