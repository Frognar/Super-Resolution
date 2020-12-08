import torch
from tqdm import tqdm


class GANTrainer:
    def __init__(self, g_net, g_criterion, g_optimizer, g_scheduler, d_net,
                 d_criterion, d_optimizer, d_scheduler, data_loader, device):
        self.g_net = g_net
        self.g_criterion = g_criterion
        self.g_optimizer = g_optimizer
        self.g_scheduler = g_scheduler

        self.d_net = d_net
        self.d_criterion = d_criterion
        self.d_optimizer = d_optimizer
        self.d_scheduler = d_scheduler

        self.data_loader = data_loader

        self.device = device

        self.start_epoch = 0
        self.losses = {
            'g_loss': list(),
            'd_loss': list()
        }

    def load_pretrained_generator(self, save_path):
        save = torch.load(save_path)
        self.g_net.load_state_dict(save['generator'])

    def load(self, save_path):
        save = torch.load(save_path)
        self.load_generator(save)
        self.load_discriminator(save)
        self.start_epoch = save['epoch']

    def load_generator(self, save):
        self.g_net.load_state_dict(save['generator'])
        self.g_optimizer.load_state_dict(save['g_optimizer'])
        self.g_scheduler.load_state_dict(save['g_scheduler'])

    def load_discriminator(self, save):
        self.d_net.load_state_dict(save['discriminator'])
        self.d_optimizer.load_state_dict(save['d_optimizer'])
        self.d_scheduler.load_state_dict(save['d_scheduler'])

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

            self.train_generator(hr_images, lr_images)
            self.train_discriminator(hr_images, lr_images)
            self.decorate_print(loop, epoch, max_epochs)

    def train_generator(self, hr_images, lr_images):
        sr_images = self.generate(lr_images)
        labels = self.discriminate(sr_images)
        loss = self.calculate_generator_loss(sr_images, hr_images, labels)
        self.optimize_generator(loss)

    def generate(self, lr_images):
        return self.g_net(lr_images)

    def discriminate(self, *images):
        return self.d_net(*images)

    def calculate_generator_loss(self, sr_images, hr_images, labels):
        loss = self.g_criterion(sr_images, hr_images, labels)
        self.losses['g_loss'].append(loss.item())
        return loss

    def optimize_generator(self, loss):
        self.g_optimizer.zero_grad()
        loss.backward()
        self.g_optimizer.step()

    def train_discriminator(self, hr_images, lr_images):
        sr_images = self.generate(lr_images).detach()
        sr_labels = self.discriminate(sr_images)
        hr_labels = self.discriminate(hr_images)
        loss = self.calculate_discriminator_loss(sr_labels, hr_labels)
        self.optimize_discriminator(loss)

    def calculate_discriminator_loss(self, sr_labels, hr_labels):
        loss = self.d_criterion(sr_labels, hr_labels)
        self.losses['d_loss'].append(loss.item())
        return loss

    def optimize_discriminator(self, loss):
        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step()

    def decorate_print(self, loop, epoch, max_epochs):
        loop.set_description(f'Epoch [{epoch + 1} / {max_epochs}]')
        mean_g_loss, mean_d_loss = self.calculate_mean_losses()
        loop.set_postfix(g_loss=mean_g_loss, d_loss=mean_d_loss)

    def calculate_mean_losses(self):
        mean_g_loss = sum(self.losses['g_loss']) / len(self.losses['g_loss'])
        mean_d_loss = sum(self.losses['d_loss']) / len(self.losses['d_loss'])
        return mean_g_loss, mean_d_loss

    def schedule_learning_rate(self):
        mean_g_loss, mean_d_loss = self.calculate_mean_losses()
        self.g_scheduler.step(mean_g_loss)
        self.d_scheduler.step(mean_d_loss)
        self.losses = {
            'g_loss': list(),
            'd_loss': list()
        }

    def save(self, epoch, save_path):
        save = dict()
        self.save_generator(save)
        self.save_discriminator(save)
        save['epoch'] = epoch + 1
        torch.save(save, save_path)

    def save_generator(self, save):
        save['generator'] = self.g_net.state_dict()
        save['g_optimizer'] = self.g_optimizer.state_dict()
        save['g_scheduler'] = self.g_scheduler.state_dict()

    def save_discriminator(self, save):
        save['discriminator'] = self.d_net.state_dict()
        save['d_optimizer'] = self.d_optimizer.state_dict()
        save['d_scheduler'] = self.d_scheduler.state_dict()
