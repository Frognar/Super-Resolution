import torch

from datasets import get_data_loader
from models import Discriminator, Generator, TruncatedVGG19
from utils.loss import PerceptualLoss


class SRGANTrainer:
    def __init__(self, train_params):
        self._initialize_models(train_params)
        self._initialize_data_loader(train_params)
        self._start_epoch = 0

    def _initialize_models(self, train_params):
        self._initialize_generator(train_params)
        self._initialize_discriminator(train_params)
        self._initialize_optimizers(train_params)
        self._initialize_criteria()

    def _initialize_generator(self, train_params):
        self._generator = Generator().cuda().train()

    def _initialize_discriminator(self, train_params):
        pool_size = self._calculate_pool_size_based_on_crop_size_and_upscale_factor(train_params)
        self._discriminator = Discriminator(pool_size=pool_size).cuda().train()

    @staticmethod
    def _calculate_pool_size_based_on_crop_size_and_upscale_factor(train_params):
        upscale_factor = 4
        return train_params['crop_size'] // (upscale_factor ** 2)

    def _initialize_optimizers(self, train_params):
        self._initialize_generator_optimizer(train_params)
        self._initialize_discriminator_optimizer(train_params)

    def _initialize_generator_optimizer(self, train_params):
        self._g_adam_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self._generator.parameters()),
            lr=train_params['learning_rate'])

    def _initialize_discriminator_optimizer(self, train_params):
        self._d_adam_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self._discriminator.parameters()),
            lr=train_params['learning_rate'])

    def _initialize_criteria(self):
        self._initialize_generator_criterion()
        self._initialize_discriminator_criterion()

    def _initialize_generator_criterion(self):
        self._g_perceptual_loss_criterion = PerceptualLoss(
            vgg=TruncatedVGG19(),
            content_criterion=torch.nn.MSELoss(),
            adversarial_criterion=torch.nn.BCELoss()
        )
        self._g_perceptual_loss_criterion.move_to_cuda()
        self._last_perceptual_loss = None

    def _initialize_discriminator_criterion(self):
        self._d_adversarial_loss_criterion = torch.nn.BCELoss().cuda()
        self._last_adversarial_loss = None

    def _initialize_data_loader(self, train_params):
        self._data_loader = get_data_loader(
            image_list_path=train_params['images_list'],
            crop_size=train_params['crop_size'],
            upscale_factor=4,
            num_workers=train_params['num_workers'],
            batch_size=train_params['batch_size']
        )

    def load_pretrained_generator(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self._generator.load_state_dict(checkpoint['generator'])

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self._load_generator(checkpoint)
        self._load_discriminator(checkpoint)
        self._start_epoch = checkpoint['epoch']

    def _load_generator(self, checkpoint):
        self._generator.load_state_dict(checkpoint['generator'])
        self._g_adam_optimizer.load_state_dict(checkpoint['g_optimizer'])

    def _load_discriminator(self, checkpoint):
        self._discriminator.load_state_dict(checkpoint['discriminator'])
        self._d_adam_optimizer.load_state_dict(checkpoint['d_optimizer'])

    def change_learning_rate(self, new_learning_rate):
        self._change_generator_learning_rate(new_learning_rate)
        self._change_discriminator_learning_rate(new_learning_rate)

    def _change_generator_learning_rate(self, new_learning_rate):
        for param_group in self._g_adam_optimizer.param_groups:
            param_group['lr'] = new_learning_rate

    def _change_discriminator_learning_rate(self, new_learning_rate):
        for param_group in self._d_adam_optimizer.param_groups:
            param_group['lr'] = new_learning_rate

    def train(self, epochs=10):
        for epoch in range(self._start_epoch, epochs):
            self._process_epoch(epoch)

    def _process_epoch(self, epoch):
        for hr_images, lr_images in self._data_loader:
            self._process_iteration(hr_images.cuda(), lr_images.cuda())
        self._save_train_checkpoint(epoch)

    def _process_iteration(self, hr_images, lr_images):
        self._train_generator(hr_images, lr_images)
        self._train_discriminator(hr_images, lr_images)

    def _train_generator(self, hr_images, lr_images):
        sr_images = self._generate_sr_images(lr_images)
        sr_discriminated = self._discriminate_images(sr_images)
        self._calculate_generator_loss(hr_images, sr_images, sr_discriminated)
        self._update_generator()

    def _generate_sr_images(self, lr_images):
        return self._generator(lr_images)

    def _discriminate_images(self, images):
        return self._discriminator(images)

    def _calculate_generator_loss(self, hr_images, sr_images, sr_discriminated):
        self._last_perceptual_loss = self._g_perceptual_loss_criterion.calculate(sr_discriminated, sr_images, hr_images)

    def _update_generator(self):
        self._g_adam_optimizer.zero_grad()
        self._last_perceptual_loss.backward()
        self._g_adam_optimizer.step()

    def _train_discriminator(self, hr_images, lr_images):
        sr_images = self._generate_sr_images(lr_images).detach()
        sr_discriminated, hr_discriminated = self._discriminate_images_for_discriminator(sr_images, hr_images)
        self._calculate_discriminator_loss(sr_discriminated, hr_discriminated)
        self._update_discriminator()

    def _discriminate_images_for_discriminator(self, sr_images, hr_images):
        sr_discriminated = self._discriminate_images(sr_images)
        hr_discriminated = self._discriminate_images(hr_images)
        return sr_discriminated, hr_discriminated

    def _calculate_discriminator_loss(self, sr_discriminated, hr_discriminated):
        sr_loss = self._d_adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated))
        hr_loss = self._d_adversarial_loss_criterion(hr_discriminated, torch.ones_like(hr_discriminated))
        self._last_adversarial_loss = sr_loss + hr_loss

    def _update_discriminator(self):
        self._d_adam_optimizer.zero_grad()
        self._last_adversarial_loss.backward()
        self._d_adam_optimizer.step()

    def _save_train_checkpoint(self, epoch):
        checkpoint = self._get_save_checkpoint_name(epoch)
        save_dict = {'generator': self._generator.state_dict(), 'g_optimizer': self._g_adam_optimizer.state_dict(),
                     'discriminator': self._discriminator.state_dict(),
                     'd_optimizer': self._d_adam_optimizer.state_dict(),
                     'epoch': epoch + 1}
        torch.save(save_dict, checkpoint)

    @staticmethod
    def _get_save_checkpoint_name(epoch):
        return f'./data/checkpoints/srgan_e{epoch + 1}.pth.tar'


class SRGANLoggerTrainer(SRGANTrainer):
    def __init__(self, train_params):
        super().__init__(train_params)
        self._print_frequency = train_params['print_frequency']

    def _process_epoch(self, epoch):
        self._losses = {'generator': list(), 'discriminator': list()}
        for iteration, (hr_images, lr_images) in enumerate(self._data_loader):
            self._process_iteration(hr_images.cuda(), lr_images.cuda())
            if self._is_time_to_print_status(iteration):
                self._print_status(iteration, epoch)
        self._save_train_checkpoint(epoch)

    def _process_iteration(self, hr_images, lr_images):
        super()._process_iteration(hr_images, lr_images)
        self._losses['generator'].append(self._last_perceptual_loss.item())
        self._losses['discriminator'].append(self._last_adversarial_loss.item())

    def _is_time_to_print_status(self, iteration):
        return iteration % self._print_frequency == 0

    def _print_status(self, iteration, epoch):
        perceptual_avg = sum(self._losses['generator']) / len(self._losses['generator'])
        adversarial_avg = sum(self._losses['discriminator']) / len(self._losses['discriminator'])
        print(f'Epoch: [{epoch + 1}] [{(iteration + 1)}/{len(self._data_loader)}]'
              f'\tGenerator loss: {perceptual_avg:.6f}\tDiscriminator loss: {adversarial_avg:.6f}')
