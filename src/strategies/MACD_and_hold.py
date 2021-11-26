from .MACD import MACD


class MACD_and_hold(MACD):
    name = 'MACD and hold'
    position_size = 1

    def exit_signal(self, _):
        return False


strategy = MACD_and_hold()
        