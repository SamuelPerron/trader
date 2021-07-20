import pandas as pd


class BigBrain:
    def __init__(self, symbol, data):
        self.symbol = symbol
        self.df = pd.DataFrame(data).set_index('t')

        self.df['RSI'] = self.get_rsi()
        self.df['MACD'] = self.get_macd()
        self.df['MACD Signal'] = self.get_macd_signal()
        self.df['5d_ma'] = self.get_ma(5)
        self.df['200d_ma'] = self.get_ma(200)


    def get_ma(self, period):
        return self.df['c'].rolling(window=period).mean()

    
    def get_rsi(self):
        period = 14
        close = self.df['c']
        delta = close.diff()
        delta = delta[1:] # Remove first entry

        # Make the positive gains (up) and negative gains (down) Series
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        # Calculate the EWMA
        roll_up1 = up.ewm(span=period).mean()
        roll_down1 = down.abs().ewm(span=period).mean()

        # Calculate the RSI based on EWMA
        rs = roll_up1 / roll_down1
        return 100.0 - (100.0 / (1.0 + rs))


    def get_macd(self):
        exp1 = self.df['c'].ewm(span=12, adjust=False).mean()
        exp2 = self.df['c'].ewm(span=26, adjust=False).mean()
        return exp1 - exp2


    def get_macd_signal(self):
        return self.df['MACD'].ewm(span=9, adjust=False).mean()