from . import Trader
from .strategies.MACD import strategy


trader = Trader(strategy)

if __name__ == '__main__':
    while True:
        trader.trade()