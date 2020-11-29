from trainers.gan_trainer import GANTrainer


class ReGANTrainer(GANTrainer):
    def discriminate(self, *images):
        return self.d_net(images[0]) - self.d_net(images[1]).detach().mean()

    def train_generator(self, real, lr_real):
        generated = self.generate(lr_real)
        labels = self.discriminate(generated, real)
        loss = self.calculate_generator_loss(real, generated, labels)
        self.update_generator(loss)

    def train_discriminator(self, real, lr_real):
        generated = self.generate(lr_real).detach()
        generated_labels = self.discriminate(generated, real)
        real_labels = self.discriminate(real, generated)
        loss = self.calculate_discriminator_loss(generated_labels, real_labels)
        self.update_discriminator(loss)


class ReGANLoggerTrainer(ReGANTrainer):
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
