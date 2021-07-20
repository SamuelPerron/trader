from .big_brain import BigBrain
import requests
import os
from dotenv import load_dotenv


load_dotenv()

# --- ENV variables --- #
LLAMA_URL = os.getenv('LLAMA_URL')
ALPACA_URL = os.getenv('ALPACA_URL')
ALPACA_UID = os.getenv('ALPACA_UID')
ALPACA_SECRET = os.getenv('ALPACA_SECRET')


class API:
    def __init__(self):
        self.llama_url = LLAMA_URL
        self.alpaca_url = ALPACA_URL


    @staticmethod
    def execute_and_log_errors(request):
        executed_request = eval(request)

        if str(executed_request.status_code)[0] != '2':
            print(f'ERROR --- {executed_request.json()}')
            # TODO: Log complete error w/ context in file

        return executed_request


    def alpaca(self, method, url, params={}, data={}):
        headers = {
            'APCA-API-KEY-ID': ALPACA_UID,
            'APCA-API-SECRET-KEY': ALPACA_SECRET
        }

        request = f'requests.{method}("{self.alpaca_url}/{url}", headers={headers}, params={params}, json={data},)'
        
        executed_request = self.execute_and_log_errors(request)

        return executed_request


    def llama(self, method, url, params={}, data={}):
        request = f'requests.{method}("{self.alpaca_url}/{url}", params={params}, json={data},)'
        
        executed_request = self.execute_and_log_errors(request)

        return executed_request


    def clock(self):
        return self.alpaca('get', 'clock').json()


    def account(self):
        return self.llama('get', 'account').json()

    
    def positions_as_symbols(self):
        return [position['symbol'] for position in self.positions()]


    def orders(self):
        # TODO: Handle filtering
        url = 'orders'
        method = 'get'

        return self.llama(method, url).json()


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

        return self.llama('post', 'orders', data=details).json()


    def positions(self, close=False):
        method = 'get'       
        url = 'positions'

        return self.llama(method, url).json()


    def close_position(self, symbol):
        method = 'post'       
        url = 'close'

        position_ids = []
        positions = self.positions()['data']
        for position in positions:
            if position['symbol'] == symbol:
                position_ids.append(position['id'])
                
        data = {
            'ids': position_ids,
        }

        return self.api('post', 'close', data=data).json()


    def bars(self, symbols, timeframe, limit=200, big_brain=False):
        params = {
            'symbols': ','.join(symbols),
            'limit': limit,
        }

        response = self.alpaca('get', f'bars/{timeframe}', params=params)
        if big_brain and response.status_code == 200:
            data = response.json()
            return [BigBrain(symbol=symbol, data=data[symbol]) for symbol in data.keys()]
        return response.json()


    def __str__(self):
        id = self.account()['id']
        return f'API <id: {id}>'
