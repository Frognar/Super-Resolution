import torch


class NetTrainer:
    def __init__(self, generator, criterion, data_loader, learning_rate, name,
                 on_cuda=True):
        self.generator = generator
        self.criterion = criterion
        self.data_loader = data_loader
        self.name = name
        self.on_cuda = on_cuda

        if self.on_cuda:
            self.move_to_cuda()

        self.optimizer = torch.optim.Adam(
            params=filter(
                lambda p: p.requires_grad,
                self.generator.parameters()
            ),
            lr=learning_rate
        )
        self.start_epoch = 0

    def move_to_cuda(self):
        self.generator = self.generator.cuda()
        self.criterion = self.criterion.cuda()

    def load(self, checkpoint):
        checkpoint = torch.load(checkpoint)
        self.generator.load_state_dict(checkpoint['generator'])
        self.optimizer.load_state_dict(checkpoint['g_optimizer'])
        self.start_epoch = checkpoint['epoch']

    def train(self, epochs=10):
        for epoch in range(self.start_epoch, epochs):
            self.process_epoch(epoch)

    def process_epoch(self, epoch):
        for real, lr_real in self.data_loader:
            if self.on_cuda:
                real, lr_real = real.cuda(), lr_real.cuda()
            self.process_iteration(real, lr_real)
        self.save(epoch)

    def process_iteration(self, real, lr_real):
        generated = self.generate(lr_real)
        loss = self.calculate_loss(generated, real)
        self.update_generator(loss)

    def generate(self, lr_images):
        return self.generator(lr_images)

    def calculate_loss(self, generated, real):
        return self.criterion(generated, real)

    def update_generator(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, epoch):
        save_dict = {
            'generator': self.generator.state_dict(),
            'g_optimizer': self.optimizer.state_dict(),
            'epoch': epoch + 1
        }
        torch.save(
            save_dict,
            f'./data/checkpoints/{self.name}_e{epoch + 1}.pth.tar'
        )


class NetLoggerTrainer(NetTrainer):
    def __init__(self, generator, criterion, data_loader, learning_rate, name,
                 logger, on_cuda=True):
        super().__init__(generator, criterion, data_loader, learning_rate,
                         name, on_cuda)
        self.logger = logger
        self.last_losses = dict()

    def process_epoch(self, epoch):
        self.logger.reset(epoch + 1)
        super().process_epoch(epoch)

    def process_iteration(self, real, lr_real):
        super().process_iteration(real, lr_real)
        self.logger.update(**self.last_losses)

    def calculate_loss(self, generated, real):
        loss = super().calculate_loss(generated, real)
        self.last_losses['Generator'] = loss.item()
        return loss
