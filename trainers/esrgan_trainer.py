import torch

from models import RRDBGenerator
from trainers.srgan_trainer import SRGANTrainer
from utils import ESRGANPerceptualLoss


class ESRGANTrainer(SRGANTrainer):
    def _initialize_generator(self, train_params):
        self._generator = RRDBGenerator().cuda().train()

    def _initialize_generator_criterion(self):
        self._g_perceptual_loss_criterion = ESRGANPerceptualLoss().cuda()
        self._last_perceptual_loss = None

    def _initialize_discriminator_criterion(self):
        self._d_adversarial_loss_criterion = torch.nn.BCEWithLogitsLoss().cuda()
        self._last_adversarial_loss = None

    def _train_generator(self, hr_images, lr_images):
        sr_images = self._generate_sr_images(lr_images)
        sr_relative_discriminated = self._relatively_discriminate_images(sr_images, hr_images)
        self._calculate_generator_loss(hr_images, sr_images, sr_relative_discriminated)
        self._update_generator()

    def _relatively_discriminate_images(self, images, comparison_images):
        discriminated_images = self._discriminate_images(images)
        discriminated_comparison_images = self._discriminate_images(comparison_images).detach()
        return discriminated_images - discriminated_comparison_images.mean()

    def _discriminate_images_for_discriminator(self, sr_images, hr_images):
        sr_discriminated = self._relatively_discriminate_images(sr_images, hr_images)
        hr_discriminated = self._relatively_discriminate_images(hr_images, sr_images)
        return sr_discriminated, hr_discriminated

    @staticmethod
    def _get_save_checkpoint_name(epoch):
        return f'./data/checkpoints/esrgan_e{epoch + 1}.pth.tar'


class ESRGANLoggerTrainer(ESRGANTrainer):
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
