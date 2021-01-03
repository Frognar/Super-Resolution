import torch
from tqdm import tqdm


class NetTrainer:
    def __init__(self, net, criterion, optimizer, scheduler, data_loader, device):
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.data_loader = data_loader

        self.device = device

        self.start_epoch = 0
        self.losses = list()

    def load(self, save_path):
        save = torch.load(save_path)
        self.net.load_state_dict(save['generator'])
        self.optimizer.load_state_dict(save['optimizer'])
        self.scheduler.load_state_dict(save['scheduler'])
        self.start_epoch = save['epoch']

    def train(self, max_epochs=20, save_path=None):
        for epoch in range(self.start_epoch, max_epochs):
            self.process_epoch(epoch, max_epochs)
            self.schedule_learning_rate()

            if save_path is not None:
                self.save(epoch, save_path)

    def process_epoch(self, epoch, max_epochs):
        loop = tqdm(self.data_loader)
        for hr_images, lr_images in loop:
            hr_images = hr_images.to(self.device)
            lr_images = lr_images.to(self.device)

            self.train_model(hr_images, lr_images)
            self.decorate_print(loop, epoch, max_epochs)

    def train_model(self, hr_images, lr_images):
        sr_images = self.generate(lr_images)
        loss = self.calculate_loss(sr_images, hr_images)
        self.optimize(loss)

    def generate(self, lr_images):
        return self.net(lr_images)

    def calculate_loss(self, sr_images, hr_images):
        loss = self.criterion(sr_images, hr_images)
        self.losses.append(loss.item())
        return loss

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decorate_print(self, loop, epoch, max_epochs):
        loop.set_description(f'Epoch [{epoch + 1} / {max_epochs}]')
        loop.set_postfix(loss=self.calculate_mean_loss())

    def calculate_mean_loss(self):
        return sum(self.losses) / len(self.losses)

    def schedule_learning_rate(self):
        self.scheduler.step(self.calculate_mean_loss())
        self.losses = list()

    def save(self, epoch, save_path):
        save = dict()
        save['generator'] = self.net.state_dict()
        save['optimizer'] = self.optimizer.state_dict()
        save['scheduler'] = self.scheduler.state_dict()
        save['epoch'] = epoch + 1
        torch.save(save, save_path)
