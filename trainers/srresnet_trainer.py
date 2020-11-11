import torch
from torch.utils.data import DataLoader

from datasets.datasets import TrainDataset
from models.models import Generator


class SRResNetTrainer:
    def __init__(self, train_params):
        self._initialize_models(train_params)
        self._initialize_data_loader(train_params)
        self._start_epoch = 0

    def _initialize_models(self, train_params):
        self._generator = Generator().cuda().train()
        self._mse_criterion = torch.nn.MSELoss().cuda()
        self._adam_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self._generator.parameters()),
                                                lr=train_params['learning_rate'])
        self._last_calculated_loss = None

    def _initialize_data_loader(self, train_params):
        dataset = TrainDataset(image_list_path=train_params['images_list'], crop_size=train_params['crop_size'],
                               upscale_factor=4)
        self._data_loader = DataLoader(dataset=dataset, num_workers=train_params['num_workers'],
                                       batch_size=train_params['batch_size'], shuffle=True,
                                       pin_memory=True)

    def load(self, checkpoint):
        checkpoint = torch.load(checkpoint)
        self._generator.load_state_dict(checkpoint['generator'])
        self._adam_optimizer.load_state_dict(checkpoint['g_optimizer'])
        self._start_epoch = checkpoint['epoch']

    def train(self, epochs=10):
        for epoch in range(self._start_epoch, epochs):
            self._process_epoch(epoch)

    def _process_epoch(self, epoch):
        for hr_images, lr_images in self._data_loader:
            self._process_iteration(hr_images.cuda(), lr_images.cuda())
        self._save_train_checkpoint(epoch)

    def _process_iteration(self, hr_images, lr_images):
        sr_images = self._generate_sr_img(lr_images)
        self._calculate_loss(sr_images, hr_images)
        self._update_generator()

    def _generate_sr_img(self, lr_images):
        return self._generator(lr_images)

    def _calculate_loss(self, sr_images, hr_images):
        self._last_calculated_loss = self._mse_criterion(sr_images, hr_images)

    def _update_generator(self):
        self._adam_optimizer.zero_grad()
        self._last_calculated_loss.backward()
        self._adam_optimizer.step()

    def _save_train_checkpoint(self, epoch):
        checkpoint = f'./data/checkpoints/srresnet_e{epoch + 1}.pth.tar'
        save_dict = {'generator': self._generator.state_dict(), 'g_optimizer': self._adam_optimizer.state_dict(),
                     'epoch': epoch + 1}
        torch.save(save_dict, checkpoint)


class SRResNetLoggerTrainer(SRResNetTrainer):
    def __init__(self, train_params):
        super().__init__(train_params)
        self._print_frequency = train_params['print_frequency']

    def _process_epoch(self, epoch):
        self._losses = list()
        for iteration, (hr_images, lr_images) in enumerate(self._data_loader):
            self._process_iteration(hr_images.cuda(), lr_images.cuda())
            if self._is_time_to_print_status(iteration):
                self._print_status(iteration, epoch)
        self._save_train_checkpoint(epoch)

    def _process_iteration(self, hr_images, lr_images):
        super()._process_iteration(hr_images, lr_images)
        self._losses.append(self._last_calculated_loss.item())

    def _is_time_to_print_status(self, iteration):
        return iteration % self._print_frequency == 0

    def _print_status(self, iteration, epoch):
        loss_avg = sum(self._losses) / len(self._losses)
        print(f'Epoch: [{epoch + 1}] [{(iteration + 1)}/{len(self._data_loader)}]\tGenerator loss: {loss_avg:.5f}')
