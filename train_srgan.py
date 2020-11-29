from torch.nn import BCELoss, MSELoss

from datasets import get_data_loader
from models import Discriminator, Generator, TruncatedVGG19
from trainers import GANLoggerTrainer
from utils import PerceptualLoss
from utils.logger import Logger


def main():
    data_loader = get_data_loader()
    srgan_trainer = GANLoggerTrainer(
        generator=Generator(),
        discriminator=Discriminator(),
        perceptual_criterion=PerceptualLoss(
            vgg=TruncatedVGG19(),
            content_criterion=MSELoss(),
            adversarial_criterion=BCELoss()
        ),
        adversarial_criterion=BCELoss(),
        data_loader=data_loader,
        learning_rate=1e-4,
        logger=Logger(
            print_frequency=508,
            max_iterations=len(data_loader)
        )
    )
    srgan_trainer.load_pretrained_generator(
        f'./data/checkpoints/srresnet_e{10}.pth.tar'
    )
    srgan_trainer.train(epochs=5)
    srgan_trainer.load(f'./data/checkpoints/srgan_e{5}.pth.tar')
    srgan_trainer.change_learning_rate(new_learning_rate=1e-5)
    srgan_trainer.train(epochs=10)


if __name__ == '__main__':
    main()
