from torch.nn import BCEWithLogitsLoss, L1Loss

from datasets import get_data_loader
from models import Discriminator, RRDBGenerator, TruncatedVGG19
from trainers import ReGANLoggerTrainer
from utils import PerceptualLoss
from utils.logger import Logger


def main():
    data_loader = get_data_loader(crop_size=128)
    esrgan_trainer = ReGANLoggerTrainer(
        generator=RRDBGenerator(),
        discriminator=Discriminator(pool_size=8),
        perceptual_criterion=PerceptualLoss(
            vgg=TruncatedVGG19(with_activation_layer=False),
            content_criterion=L1Loss(),
            adversarial_criterion=BCEWithLogitsLoss()
        ),
        adversarial_criterion=BCEWithLogitsLoss(),
        data_loader=data_loader,
        learning_rate=1e-4,
        logger=Logger(
            print_frequency=508,
            max_iterations=len(data_loader)
        )
    )
    esrgan_trainer.load_pretrained_generator(
        f'./data/checkpoints/srrrdbnet_e{10}.pth.tar'
    )
    esrgan_trainer.train(epochs=5)
    esrgan_trainer.load(f'./data/checkpoints/esrgan_e{5}.pth.tar')
    esrgan_trainer.change_learning_rate(new_learning_rate=1e-5)
    esrgan_trainer.train(epochs=10)


if __name__ == '__main__':
    main()
