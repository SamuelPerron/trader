import sys

from src.backtests.backtest_bot import BacktestBot

if __name__ == '__main__':
    symbol = sys.argv[1] 
    strategy = sys.argv[2] 
    interval = sys.argv[3] 

    if len(sys.argv) > 4:
        starting_capital = sys.argv[4]
    else:
        starting_capital = 1000

    if len(sys.argv) > 5:
        money_added_at_interval = sys.argv[5]
    else:
        money_added_at_interval = 0

    BacktestBot(
        symbol, 
        strategy,
        interval,
        starting_capital,
        money_added_at_interval,
    )