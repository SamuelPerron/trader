from .MACD import MACD


class MACD(MACD):
    def __init__(self):
        super().__init__()

        self.name = 'MACD and hold'
        self.position_size = 1

    def exit_signal(self, _):
        return False


strategy = MACD()
        