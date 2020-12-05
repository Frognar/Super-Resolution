from trainers.gan_trainer import GANTrainer


class ReGANTrainer(GANTrainer):
    def discriminate(self, *images):
        i, c = images
        return self.d_net(i) - self.d_net(c).detach().mean()

    def train_generator(self, hr_images, lr_images):
        sr_images = self.generate(lr_images)
        labels = self.discriminate(sr_images, hr_images)
        loss = self.calculate_generator_loss(sr_images, hr_images, labels)
        self.optimize_generator(loss)

    def train_discriminator(self, hr_images, lr_images):
        sr_images = self.generate(lr_images).detach()
        sr_labels = self.discriminate(sr_images, hr_images)
        hr_labels = self.discriminate(hr_images, sr_images)
        loss = self.calculate_discriminator_loss(sr_labels, hr_labels)
        self.optimize_discriminator(loss)
