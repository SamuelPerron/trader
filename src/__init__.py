from .api import API
from .utils import log
from datetime import datetime
import csv
import os
import time


class Trader:
    CSV_FILE = 'current_symbols.csv'

    def __init__(self, strategy):
        self.api = API()
        self.strategy = strategy
        self.timeframe = 'minute'
        self.symbols = []


    def sleep(self, sleep_time=60):
        if sleep_time >= 60:
            log(f'Sleeping for {round(sleep_time / 60)} minutes...')

        else:
            log(f'Sleeping for {sleep_time} seconds...')

        time.sleep(sleep_time)


    def trade(self):
        now = datetime.now()
        log(now)

        clock = self.api.clock()
        next_open = datetime.strptime(clock['next_open'][:-6], '%Y-%m-%dT%H:%M:%S')

        if clock['is_open']:
            self.health_print(now)

            if not os.path.exists(Trader.CSV_FILE):
                self.find_next_symbols()
            
            self.fetch_symbols()

            symbols_bars = self.api.bars(
                self.symbols, timeframe=self.timeframe, big_brain=True
            )

            for symbol_bars in symbols_bars:
                if self.strategy.check_for_entry_signal(symbol_bars.df):
                    self.buy(symbol_bars.symbol, symbol_bars.df['c'])

                if self.strategy.check_for_exit_signal(symbol_bars.df):
                    self.sell(symbol_bars.symbol)
                    
            self.sleep(self.strategy.sleep)

        elif (next_open - now).total_seconds() <= 120:
            self.find_next_symbols()
            log(f'--- NEW SYMBOLS | {now.strftime("%Y-%m-%d")} ---\n{", ".join(self.symbols)}')
            self.sleep()

        else:
            next_open_minutes = round(
                (next_open - now).total_seconds() / 60, 0
            )

            if next_open_minutes < 60:
                log(f'Markets open in {next_open_minutes} minutes.')
                sleep_time = 60 * 20

            else:
                log('Markets closed.')
                sleep_time = 60 * 60

            self.remove_symbols()
            self.sleep(sleep_time)


    def buy(self, symbol, price):
        if len(
            self.api.orders(filters={'status': 'open', 'symbols': symbol})
        ) == 0:
        
            price = price.iloc[-1]
            stop_loss = self.strategy.find_stop_loss(price)
            buying_power = float(self.api.account()['buying_power'])
            qty = self.strategy.find_qty(price, buying_power)

            if qty > 0:
                order = {
                    'symbol': symbol,
                    'side': 'buy',
                    'type': 'market',
                    'qty': qty,
                    'order_class': 'oto',
                    'stop_loss': {
                        'stop_price': stop_loss,
                        'limit_price': stop_loss - 0.04
                    }
                }
                
                self.api.new_order(order)
                log(f'--- BUY ORDER ---\n    {symbol} x{qty} @ $ {round(float(price), 2)}')


    def sell(self, symbol):
        if symbol in self.api.positions_as_symbols():
            self.api.close_position(symbol)
            log(f'--- CLOSING POSITION ---\n    {symbol}')


    def find_next_symbols(self):
        strategy_symbols = self.strategy.find_next_symbols()
        positions_symbols = [position['symbol'] for position in self.api.positions()]
        symbols = strategy_symbols + positions_symbols

        self.create_symbols_file()
        with open(Trader.CSV_FILE, mode='w') as file:
            writer = csv.writer(file)
            for symbol in symbols:
                writer.writerow([symbol,])
        

    def remove_symbols(self):
        if os.path.exists(Trader.CSV_FILE):
            os.remove(Trader.CSV_FILE)


    def fetch_symbols(self):
        with open(Trader.CSV_FILE, mode='r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                self.symbols.append(row[0])
            

    def create_symbols_file(self):
        open(Trader.CSV_FILE, 'w')

        
    def health_print(self, now):
        """
        Logs current status of the account in the console
        """

        log(f'\n\n{now.strftime("%Y-%m-%d %H:%M:%S")}')

        account = self.api.account()
        last_equity = account['last_equity']
        equity = account['equity']
        pl = round(
            (
                ((equity * 100) / last_equity) - 100
            ), 2
        )

        log(f'BP: $ {account["buying_power"]} | PV: $ {equity} | P/L: {pl}%')