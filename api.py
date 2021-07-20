from .big_brain import BigBrain
import requests


class API:
    def __init__(self):
        self.base_url = 'https://paper-api.alpaca.markets/v2'
        self.market_base_url = 'https://data.alpaca.markets/v1'


    def api(self, method, url, params={}, data={}):
        headers = {
            'APCA-API-KEY-ID': API_KEY,
            'APCA-API-SECRET-KEY': API_SECRET
        }
        base_url = self.base_url
        if 'bars' in url:
            base_url = self.market_base_url
        request = f'requests.{method}("{base_url}/{url}", headers={headers}, params={params}, json={data},)'
        executed_request = eval(request)

        if str(executed_request.status_code)[0] != '2':
            print(f'ERROR --- {executed_request.json()}')
            # TODO: Log complete error w/ context in file
        return executed_request


    def clock(self):
        return self.api('get', 'clock').json()


    def account(self):
        return self.api('get', 'account').json()

    
    def positions_as_symbols(self):
        return [position['symbol'] for position in self.positions()]


    def orders(self, id=None, filters={}, cancel=False):
        possible_filters = (
            'status', 'limit', 'after', 
            'until', 'direction', 'nested', 'symbols'
        )
        for ftr in filters.keys():
            if ftr not in possible_filters:
                print(f'ERROR --- `{ftr}` is not an acceptable filter.')
        
        if id and filters != {}:
            print(f'ERROR --- Can\'t filter when getting a specific order.')

        url = 'orders'
        if id:
            url += f'/{id}'

        method = 'get'
        if cancel:
            method = 'delete'

        return self.api(method, url, filters).json()


    def new_order(self, details):
        details.update({'time_in_force': 'gtc'})
        required_fields = ('symbol', 'qty', 'side', 'type')

        type = details.get('type')
        if type in ('limit', 'stop_limit'):
            required_fields.append('limit_price')
        if type in ('stop', 'stop_limit', 'trailing_stop'):
            required_fields.append('stop_price')

        for field in required_fields:
            if not details.get(field):
                print(f'ERROR --- `{field}` is required.')

        return self.api('post', 'orders', data=details).json()


    def positions(self, symbol=None, close=False):
        method = 'get'
        if close:
            method = 'delete'
        
        url = 'positions'
        if symbol:
            url += f'/{symbol}'

        return self.api(method, url).json()


    def bars(self, symbols, timeframe, limit=200, big_brain=False):
        params = {
            'symbols': ','.join(symbols),
            'limit': limit,
        }

        response = self.api('get', f'bars/{timeframe}', params=params)
        if big_brain and response.status_code == 200:
            data = response.json()
            return [BigBrain(symbol=symbol, data=data[symbol]) for symbol in data.keys()]
        return response.json()


    def __str__(self):
        id = self.account()['id']
        return f'API <id: {id}>'
