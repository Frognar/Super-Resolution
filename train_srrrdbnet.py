from torch.nn import MSELoss

from datasets import get_data_loader
from models import RRDBGenerator
from trainers import NetLoggerTrainer
from utils.logger import Logger


def main():
    data_loader = get_data_loader(crop_size=128)
    srrrdbnet_trainer = NetLoggerTrainer(
        generator=RRDBGenerator(),
        criterion=MSELoss(),
        data_loader=data_loader,
        learning_rate=1e-4,
        name='srrrdbnet',
        logger=Logger(
            print_frequency=508,
            max_iterations=len(data_loader)
        ),
        on_cuda=True
    )
    srrrdbnet_trainer.train(epochs=10)


if __name__ == '__main__':
    main()
