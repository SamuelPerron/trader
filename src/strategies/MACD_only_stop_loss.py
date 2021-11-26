from .MACD import MACD


class MACD_only_stop_loss(MACD):
    name = 'MACD only stop loss'
    stop_loss = 0.021
    stop_loss_and_sell_signal = False

    def exit_signal(self, row):
        return False


strategy = MACD_only_stop_loss()
        