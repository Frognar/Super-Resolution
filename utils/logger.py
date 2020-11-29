class Logger:
    def __init__(self, print_frequency, max_iterations):
        self.print_frequency = print_frequency
        self.iteration = 0
        self.max_iterations = max_iterations
        self.epoch = 0
        self.losses = dict()

    def reset(self, epoch):
        self.epoch = epoch
        self.losses = dict()

    def is_time_to_print(self):
        return self.iteration % self.print_frequency == 0

    def update(self, **loss):
        for name in loss:
            if name not in self.losses:
                self.losses[name] = list()
            self.losses[name].append(loss[name])

        self.iteration += 1
        if self.is_time_to_print():
            self.print_log()

    def print_log(self):
        avgs = dict()
        for key in self.losses.keys():
            avgs[key] = (sum(self.losses[key]) / len(self.losses[key]))
        print(f'[{self.epoch}][{self.iteration}/{self.max_iterations}] '
              f'{[f"{key} loss: {avgs[key]:.6f}" for key in avgs]}')
