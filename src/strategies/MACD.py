from . import Strategy
from ..screeners import MarketWatch


class MACD(Strategy):
    def __init__(self):
        self.name = 'MACD'
        self.sleep = 60
        self.position_size = 0.15
        self.stop_loss = 0.021
        self.take_profit = 0.02
        self.overbought_rsi = 70
        self.macd_crossover_threshold = 0.02


    def check_for_entry_signal(self, data, *args, **kwargs):
        data_point = data.iloc[-1]
        return self.entry_signal(data_point)


    def entry_signal(self, row):
        buy = False
        crossover, below_zero = self.is_macd_crossover(row['MACD'], row['MACD Signal'])

        crossover, below_zero = self.is_macd_crossover(row['MACD'], row['MACD Signal'])
        if crossover and\
        below_zero and\
        row['5d_ma'] > row['200d_ma']:
            buy = True

        return buy

    
    def check_for_exit_signal(self, data, *args, **kwargs):
        data_point = data.iloc[-1]
        return self.exit_signal(data_point)


    def exit_signal(self, row):
        sell = False

        crossover, below_zero = self.is_macd_crossover(row['MACD'], row['MACD Signal'])
        if crossover and\
        not below_zero and\
        row['RSI'] >= self.overbought_rsi:
            sell = True

        return sell


    def is_macd_crossover(self, macd, signal):
        """
        Returns if a crossover occured 
        and if it was below the 0 line.
        """
        return (
            abs(macd) - abs(signal)
        ) <= self.macd_crossover_threshold, macd < 0 and signal < 0


    def find_next_symbols(self):
        mw = MarketWatch()
        return [symbol['symbol'] for symbol in mw.pre_market()['most_actives']]


    def find_qty(self, price, buying_power):
        perc_capital = buying_power * self.position_size
        return int(round(perc_capital / price, 0))


    def find_take_profit(self, price):
        return price * (1 + self.take_profit)


    def find_stop_loss(self, price):
        return price * (1 - self.stop_loss)


strategy = MACD()
        