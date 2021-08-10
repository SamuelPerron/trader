class Strategy:
    def check_for_entry_signal(self, data):
        raise NotImplementedError()


    def check_for_exit_signal(self, data):
        raise NotImplementedError()

    
    def find_next_symbols(self):
        raise NotImplementedError()