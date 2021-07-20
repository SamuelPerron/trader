import requests
from bs4 import BeautifulSoup


class MarketWatch:
    base_url = 'https://www.marketwatch.com/tools/screener'

    @staticmethod
    def vol_to_float(vol):
        if 'K' in vol:
            vol = float(vol[:-1]) * 1000
        elif 'M' in vol:
            vol = float(vol[:-1]) * 1000000
        else:
            vol = float(vol.replace(',', ''))
        return vol


    def pre_market(self):
        html = requests.get(f'{self.base_url}/premarket').text
        soup = BeautifulSoup(html, 'html.parser')
        categories = soup.find_all('div', {'class': 'group--elements'})
        categories_matches = {
            'Leaders': 'gainers',
            'Laggards': 'loosers',
            'Most Active': 'most_actives',
        }
        results = {}
        
        for category in categories:
            title = category.find('h2', {'class': 'title'}).text.strip()
            results[categories_matches[title]] = []

            for stock in category.find('tbody').find_all('tr', {'class': 'table__row'}):
                cells = stock.find_all('td')
                results[categories_matches[title]].append({
                    'symbol': cells[0].text.strip(),
                    'pre_price': MarketWatch.vol_to_float(cells[2].text.strip()[1:]),
                    'pre_volume': MarketWatch.vol_to_float(cells[3].text.strip()),
                    'pre_change': float(cells[4].text.strip()),
                    'pre_change_perc': float(cells[5]
                                        .find('li', {'class': 'content__item'})
                                        .text.strip()[:-1])
                })

        return results