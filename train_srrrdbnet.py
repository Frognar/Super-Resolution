from trainers import SRRRDBNetLoggerTrainer
from utils import get_training_params


def main():
    srrrdbnet_params = get_training_params(crop_size=128)
    srrrdbnet_trainer = SRRRDBNetLoggerTrainer(srrrdbnet_params)
    srrrdbnet_trainer.train(epochs=10)


if __name__ == '__main__':
    main()
