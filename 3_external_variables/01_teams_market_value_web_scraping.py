from bs4 import BeautifulSoup
import requests as req
import pandas as pd

SEASONS = [2018, 2019, 2020, 2021]

# code to find league - ES - Espanyol, 2 - second league level
CODE = 'ES2'

# link that works for all leagues (change only code & season for request)
# url = f'https://www.transfermarkt.com/laliga2/startseite/wettbewerb/{code}/plus/?saison_id={season}'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}

data = {'team': [],
        'season': [],
        'squad': [],
        'age': [],
        'foreigners': [],
        'market_value': [],
        'total_market_value': []}

# web scraping
for season in SEASONS:
    url = f'https://www.transfermarkt.com/laliga2/startseite/wettbewerb/{CODE}/plus/?saison_id={SEASONS}'

    response = req.get(url, headers=HEADERS)
    bs = BeautifulSoup(response.content, 'html.parser')

    all_tr = bs.find_all('tr', {'class': ['odd', 'even']}, limit=22)  # limit - number of teams in the league

    for row in all_tr:
        all_td = row.find_all('td', recursive=False)

        data['team'].append(all_td[1].text)
        data['squad'].append(all_td[2].text)
        data['season'].append(season)
        data['age'].append(all_td[3].text)
        data['foreigners'].append(all_td[4].text)
        data['market_value'].append(all_td[5].text)
        data['total_market_value'].append(all_td[6].text)

# create dataframe
df = pd.DataFrame(data)


def convert_currency_to_number(df_row):
    """Convert currency values format to number"""
    if df_row[-1] == 'm':
        return df_row.replace('.', '')[1:-1] + '0000'
    else:
        return df_row[1:-3] + '000'


# clear strings
df['team'] = df['team'].apply(lambda row: row.strip())

# converting value columns
df['market_value'] = df['market_value'].apply(convert_currency_to_number)
df['total_market_value'] = df['total_market_value'].apply(convert_currency_to_number)

# save to csv
df.to_csv('team_market_value')

if __name__ == '__main__':
    print(df.head())
    df.to_csv('team_market_value_web_scraping_2.csv', index=False)
