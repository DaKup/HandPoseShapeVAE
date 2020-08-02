from tensorboardX import SummaryWriter

class Logger(SummaryWriter):

    def __init__(self, model: str, log_interval=10, log_dir = None, basename ='', **kwargs):
        SummaryWriter.__init__(self, log_dir, basename, **kwargs)

        self.epoch = 0
        self.step = 0
        self.log_interval = log_interval
        self.basename = basename
        self.model = model

    def next_step(self):
        self.step += 1

    def next_epoch(self):
        self.epoch += 1
