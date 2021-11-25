from datetime import datetime, timedelta
import re
import pandas as pd
import numpy as np

from typing import Optional
from src.strategies import Strategy
from src.utils import CoolEnum
from src.big_brain import BigBrain
from src.strategies.MACD import MACD


class IntervalUnitChoices(CoolEnum):
    DAY = ('d', 'days')
    WEEK = ('w', 'weeks')


class BacktestBot:
    """
    This takes a symbol, a strategy and some other parameters and starts a 
    backtest with the available historical data.

    It'll feed the whole dataset to the big_brain module, which will output
    a Panda dataframe with multiple signals already computed. This dataframe
    will then be used to compute the final dataframe, the one that is going
    to register the buy and sell signals. With this final dataframe, we'll
    be able to evaluate the strategy.
    """

    def __init__(
        self, 
        symbol: str, 
        strategy: Strategy,
        interval: str,
        starting_capital: Optional[float] = 1000,
        money_added_at_interval: Optional[float] = 0,
    ) -> None:
        """
        The interval string is built like this:
            10m
        "10" being the number of unit
        "m" being the unit, in this case minutes

        So in this example, the trader would exercise it's strategy 
        every 10 minutes.
        """

        self.symbol = symbol
        self.strategy = MACD()
        self.interval = self.compute_interval_to_timedelta(interval)
        self.starting_capital = starting_capital
        self.money_added_at_interval = money_added_at_interval

        self.df = self.load_df()
        print(self.df)

    @staticmethod
    def compute_interval_to_timedelta(interval: str) -> timedelta:
        unit = interval[-1]
        if unit not in IntervalUnitChoices.values_short():
            raise ValueError(
                'The unit for the interval is incorrect.'
            )

        try:
            period = int(re.search('^[0-9]*', interval).group())
        except ValueError:
            raise ValueError(
                'You must specify a number for the interval.'
            )

        minutes = 0
        if unit == 'm':
            minutes = period
        elif unit == 'h':
            minutes = period * 60
        elif unit == 'd':
            minutes = period * 60 * 24
        elif unit == 'w':
            minutes = period * 60 * 24 * 7

        return timedelta(minutes=minutes)

    def load_df(self) -> pd.DataFrame:
        # Read and interpret the file
        path = f'./src/backtests/historical_data/{self.symbol}.csv'
        df = pd.read_csv(
            path, 
            names=('t', 'o', 'h', 'l', '-', 'c', 'v')
        )[1:]
        del df['-']
        df['t'] = df['t'].apply(pd.to_datetime)
        df['o'] = df['o'].apply(pd.to_numeric)
        df['h'] = df['h'].apply(pd.to_numeric)
        df['l'] = df['l'].apply(pd.to_numeric)
        df['c'] = df['c'].apply(pd.to_numeric)
        df['v'] = df['v'].apply(pd.to_numeric)
        df = df.set_index('t')

        # Feed it to BigBrain
        df = BigBrain(symbol=self.symbol, df=df).df

        # Add invested capital
        df = self._add_invested_capital(df)

        # Add buy and sell signals
        df = self._add_buy_signals(df)
        df = self._add_sell_signals(df)

        return df

    def _add_invested_capital(self, df: pd.DataFrame) -> pd.DataFrame:
        start, end = df.index[0], df.index[-1]
        df['invested_capital_movement'] = 0
        df.loc[start, 'invested_capital_movement'] = self.starting_capital

        if self.money_added_at_interval == 0:
            df['invested_capital'] = self.starting_capital
            return df

        def calculate_invested_capital(invested_capital: list, movement: list) -> list:
            res = np.empty(invested_capital.shape)
            res[0] = invested_capital[0]
            for i in range(1, invested_capital.shape[0]):
                res[i] = res[i-1] + float(movement[i])
            return res

        dates = self.get_dates_from_intervals(start, end)
        df.loc[df.index.isin(dates), 'invested_capital_movement'] = self.money_added_at_interval
        df.loc[start, 'invested_capital'] = self.starting_capital
        df['invested_capital'] = calculate_invested_capital(
            df['invested_capital'].values.T,
            df['invested_capital_movement'].values.T,
        )

        return df

    def _add_buy_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df['buy'] = df.apply(lambda row: self.strategy.entry_signal(row), axis=1)
        return df

    def _add_sell_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df['exit'] = df.apply(lambda row: self.strategy.exit_signal(row), axis=1)
        return df

    def get_dates_from_intervals(self, start: datetime, end: datetime) -> list:
        dates = []
        current_date = start
        while current_date < end:
            current_date = current_date + self.interval
            dates.append(current_date.strftime('%Y-%m-%d'))

        return dates