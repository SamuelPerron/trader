class Strategy:
    name = 'Strategy'
    sleep = 0
    stop_loss_and_sell_signal = False # RISKY
    position_size = 0
    stop_loss = 0
    take_profit = 0

    def check_for_entry_signal(self, data):
        raise NotImplementedError()


    def check_for_exit_signal(self, data):
        raise NotImplementedError()

    
    def find_next_symbols(self):
        raise NotImplementedError()