from trainers import SRResNetLoggerTrainer
from utils import get_training_params


def main():
    srresnet_params = get_training_params()
    srresnet_trainer = SRResNetLoggerTrainer(srresnet_params)
    srresnet_trainer.train(epochs=10)


if __name__ == '__main__':
    main()
