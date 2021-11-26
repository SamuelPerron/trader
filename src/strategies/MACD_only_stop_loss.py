from .MACD import MACD


class MACD_only_stop_loss(MACD):
    name = 'MACD only stop loss'
    stop_loss = 0.021

    def exit_signal(self, row):
        return True


strategy = MACD_only_stop_loss()
        