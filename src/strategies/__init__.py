class Strategy:
    name = 'Strategy'
    sleep = None
    stop_loss_and_sell_signal = False # RISKY
    position_size = None
    stop_loss = None
    take_profit = None

    def check_for_entry_signal(self, data):
        raise NotImplementedError()


    def check_for_exit_signal(self, data):
        raise NotImplementedError()

    
    def find_next_symbols(self):
        raise NotImplementedError()