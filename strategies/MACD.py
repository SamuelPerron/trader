from alpaca_paper.strategies import Strategy
from alpaca_paper.screeners import MarketWatch


class MACD(Strategy):
    def __init__(self):
        self.name = 'MACD'
        self.position_size = 0.15
        self.stop_loss = 0.025
        self.take_profit = 0.02
        self.overbought_rsi = 70
        self.macd_crossover_threshold = 0.02


    def check_for_entry_signal(self, data, *args, **kwargs):
        buy = False
        data_point = data.iloc[-1]

        crossover, below_zero = self.is_macd_crossover(data_point['MACD'], data_point['MACD Signal'])
        if crossover and\
        below_zero and\
        data_point['5d_ma'] > data_point['200d_ma']:
            buy = True

        return buy

    
    def check_for_exit_signal(self, data, *args, **kwargs):
        sell = False
        data_point = data.iloc[-1]

        crossover, below_zero = self.is_macd_crossover(data_point['MACD'], data_point['MACD Signal'])
        if crossover and\
        not below_zero and\
        data_point['RSI'] >= self.overbought_rsi:
            sell = True

        return sell


    def is_macd_crossover(self, macd, signal):
        """
        Returns if a crossover occured 
        and if it was below the 0 line.
        """
        return (abs(macd) - abs(signal)) <= self.macd_crossover_threshold, macd < 0 and signal < 0


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
        