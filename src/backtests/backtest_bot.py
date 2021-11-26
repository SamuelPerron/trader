from datetime import datetime, timedelta
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

        self.df.plot(y=['portfolio_worth', 'c', 'invested_capital', 'stop_loss'], kind = 'line')
        self.df.to_csv('df.csv', index=True)

        last_row = self.df.iloc[-1]
        print(f"""
        RESULTS
        -------

        ROI: {(abs(last_row['invested_capital'] - last_row['portfolio_worth']) / ((last_row['invested_capital'] + last_row['portfolio_worth']) / 2)) * 100}%
        Total invested: {last_row['invested_capital']}$
        Portfolio worth: {last_row['portfolio_worth']}$
        """)

        plt.show()

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
        df['buy'] = df.apply(lambda row: self.strategy.entry_signal(row), axis=1)
        df['sell'] = df.apply(lambda row: self.strategy.exit_signal(row), axis=1)

        # Add portfolio investments
        df = self._invest(df)

        return df

    def _add_invested_capital(self, df: pd.DataFrame) -> pd.DataFrame:
        start, end = df.index[0], df.index[-1]
        df['invested_capital_movement'] = 0
        df.loc[start, 'invested_capital_movement'] = self.starting_capital

        if self.money_added_at_interval == 0:
            df['invested_capital'] = self.starting_capital
            return df

        def calculate_invested_capital(invested_capital: list, movement: list) -> list:
            result = np.empty(invested_capital.shape)
            result[0] = invested_capital[0]
            for i in range(1, invested_capital.shape[0]):
                result[i] = result[i-1] + float(movement[i])
            return result

        dates = self.get_dates_from_intervals(start, end)
        df.loc[df.index.isin(dates), 'invested_capital_movement'] = self.money_added_at_interval
        df.loc[start, 'invested_capital'] = self.starting_capital
        df['invested_capital'] = calculate_invested_capital(
            df['invested_capital'].values.T,
            df['invested_capital_movement'].values.T,
        )

        return df

    def _invest(self, df):
        def get_available_capital_and_quantities(
            capital, 
            price, 
            buy, 
            sell, 
            invested_capital_movement,
        ) -> list:
            available_capital = np.empty(capital.shape)
            quantities = np.empty(capital.shape)
            stop_loss = np.empty(capital.shape)

            available_capital[0] = capital[0]
            quantities[0] = 0
            stop_loss[0] = 0

            running_quantities = []
            running_stop_loss = {
                'sum': 0,
                'nb_stops': 0,
            }
            for i in range(1, capital.shape[0]):
                stop_loss_condition = (
                    running_stop_loss['nb_stops'] > 0
                    and (running_stop_loss['sum'] / running_stop_loss['nb_stops']) <= price[i]
                )

                sell_condition = (
                    stop_loss_condition and sell[i]
                ) if self.strategy.stop_loss_and_sell_signal else (
                    stop_loss_condition or sell[i]
                )

                if buy[i]:
                    qty = self.strategy.find_qty(price[i], available_capital[i-1])
                    if qty != 0:
                        running_quantities.append(qty)
                        quantities[i] = qty
                        price_of_buy = running_quantities[-1] * price[i]
                        available_capital[i] = available_capital[i-1] - price_of_buy
                        running_stop_loss['sum'] += self.strategy.find_stop_loss(
                            price[i]
                        )
                        running_stop_loss['nb_stops'] += 1
                    
                    else:
                        available_capital[i] = available_capital[i-1]
                        quantities[i] = 0

                elif (
                    len(running_quantities) != 0
                    and sell_condition
                ):
                    roi = sum(running_quantities) * price[i]
                    available_capital[i] = available_capital[i-1] + roi
                    quantities[i] = -sum(running_quantities)
                    running_quantities = []
                    running_stop_loss['sum'] = 0
                    running_stop_loss['nb_stops'] = 0

                else:
                    available_capital[i] = available_capital[i-1]
                    quantities[i] = 0

                available_capital[i] += float(invested_capital_movement[i])

                stop_loss[i] = running_stop_loss['sum']

            return available_capital, quantities, stop_loss

        def calculate_portfolio_worth(available_capital, price, quantities) -> list:
            portfolio_worth = np.empty(available_capital.shape)
            portfolio_worth[0] = available_capital[0]
            running_quantities = 0
            for i in range(1, available_capital.shape[0]):
                running_quantities += quantities[i]
                portfolio_worth[i] = available_capital[i] + (running_quantities * price[i])

            return portfolio_worth

        df.loc[df.index[0], 'available_capital'] = self.starting_capital
        df['available_capital'], df['quantities'], df['stop_loss'] = get_available_capital_and_quantities(
            df['available_capital'].values.T,
            df['c'].values.T,
            df['buy'].values.T,
            df['sell'].values.T,
            df['invested_capital_movement'].values.T,
        )

        df['portfolio_worth'] = calculate_portfolio_worth(
            df['available_capital'].values.T,
            df['c'].values.T,
            df['quantities'].values.T,
        )

        return df

    def get_dates_from_intervals(self, start: datetime, end: datetime) -> list:
        dates = []
        current_date = start
        while current_date < end:
            current_date = current_date + self.interval
            dates.append(current_date.strftime('%Y-%m-%d'))

        return dates
