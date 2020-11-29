import torch


class GANTrainer:
    def __init__(self, generator, discriminator, perceptual_criterion,
                 adversarial_criterion, data_loader, learning_rate, name,
                 on_cuda=True):
        self.on_cuda = on_cuda
        self.name = name

        self.g_net = generator
        self.d_net = discriminator

        self.perceptual_criterion = perceptual_criterion
        self.adversarial_criterion = adversarial_criterion

        if self.on_cuda:
            self.move_to_cuda()

        self.g_optimizer = self.create_optimizer(self.g_net, learning_rate)
        self.d_optimizer = self.create_optimizer(self.d_net, learning_rate)

        self.data_loader = data_loader
        self.start_epoch = 0

    def move_to_cuda(self):
        self.g_net = self.g_net.cuda()
        self.d_net = self.d_net.cuda()
        self.perceptual_criterion.move_to_cuda()
        self.adversarial_criterion = self.adversarial_criterion.cuda()

    @staticmethod
    def create_optimizer(model, learning_rate):
        return torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate
        )

    def load_pretrained_generator(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.g_net.load_state_dict(checkpoint['generator'])

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.g_net.load_state_dict(checkpoint['generator'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        self.d_net.load_state_dict(checkpoint['discriminator'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        self.start_epoch = checkpoint['epoch']

    def change_learning_rate(self, new_learning_rate):
        self.set_new_learning_rate(self.g_optimizer, new_learning_rate)
        self.set_new_learning_rate(self.d_optimizer, new_learning_rate)

    @staticmethod
    def set_new_learning_rate(optimizer, new_learning_rate):
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_learning_rate

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
        self.train_generator(real, lr_real)
        self.train_discriminator(real, lr_real)

    def train_generator(self, real, lr_real):
        generated = self.generate(lr_real)
        labels = self.discriminate(generated)
        loss = self.calculate_generator_loss(real, generated, labels)
        self.update_generator(loss)

    def generate(self, lr_images):
        return self.g_net(lr_images)

    def discriminate(self, *images):
        return self.d_net(*images)

    def calculate_generator_loss(self, real, generated, labels):
        return self.perceptual_criterion.calculate(labels, generated, real)

    def update_generator(self, loss):
        self.g_optimizer.zero_grad()
        loss.backward()
        self.g_optimizer.step()

    def train_discriminator(self, real, lr_real):
        generated = self.generate(lr_real).detach()
        generated_labels = self.discriminate(generated)
        real_labels = self.discriminate(real)
        loss = self.calculate_discriminator_loss(generated_labels, real_labels)
        self.update_discriminator(loss)

    def calculate_discriminator_loss(self, generated_labels, real_labels):
        sr_loss = self.adversarial_criterion(
            generated_labels,
            torch.zeros_like(generated_labels)
        )
        hr_loss = self.adversarial_criterion(
            real_labels,
            torch.ones_like(real_labels)
        )
        return sr_loss + hr_loss

    def update_discriminator(self, loss):
        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step()

    def save(self, epoch):
        save_dict = {
            'generator': self.g_net.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'discriminator': self.d_net.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'epoch': epoch + 1
        }
        torch.save(
            save_dict,
            f'./data/checkpoints/{self.name}_e{epoch + 1}.pth.tar'
        )


class GANLoggerTrainer(GANTrainer):
    def __init__(self, generator, discriminator, perceptual_criterion,
                 adversarial_criterion, data_loader, learning_rate,
                 logger, on_cuda=True):
        super().__init__(generator, discriminator, perceptual_criterion,
                         adversarial_criterion, data_loader, learning_rate,
                         on_cuda)
        self.logger = logger
        self.last_losses = dict()

    def process_epoch(self, epoch):
        self.logger.reset(epoch + 1)
        super().process_epoch(epoch)

    def process_iteration(self, real, lr_real):
        super().process_iteration(real, lr_real)
        self.logger.update(**self.last_losses)

    def calculate_generator_loss(self, real, generated, labels):
        g_loss = super().calculate_generator_loss(real, generated, labels)
        self.last_losses['Generator'] = g_loss.item()
        return g_loss

    def calculate_discriminator_loss(self, generated_labels, real_labels):
        d_loss = super().calculate_discriminator_loss(
            generated_labels,
            real_labels
        )
        self.last_losses['Discriminator'] = d_loss.item()
        return d_loss
