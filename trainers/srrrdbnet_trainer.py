from models import RRDBGenerator
from trainers import SRResNetTrainer


class SRRRDBNetTrainer(SRResNetTrainer):
    def _initialize_generator(self):
        self._generator = RRDBGenerator().cuda().train()

    @staticmethod
    def _get_save_checkpoint_name(epoch):
        return f'./data/checkpoints/srrrdbnet_e{epoch + 1}.pth.tar'


class SRRRDBNetLoggerTrainer(SRRRDBNetTrainer):
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
