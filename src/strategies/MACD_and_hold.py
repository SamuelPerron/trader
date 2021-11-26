from .MACD import MACD


class MACD_and_hold(MACD):
    name = 'MACD and hold'
    position_size = 1
    stop_loss_and_sell_signal = True

    def exit_signal(self, _):
        return False


strategy = MACD_and_hold()
        