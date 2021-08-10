from . import Trader
from .strategies.MACD import strategy


if __name__ == '__main__':
    Trader(strategy)