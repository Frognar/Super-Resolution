from trainers import SRGANLoggerTrainer
from utils import get_gan_training_params


def main():
    srgan_params = get_gan_training_params()
    srgan_trainer = SRGANLoggerTrainer(srgan_params)
    train_pretrained_model(srgan_trainer)
    train_model_with_adjusted_learning_rate(srgan_trainer, new_learning_rate=srgan_params['new_learning_rate'])


def train_pretrained_model(srgan_trainer):
    srgan_trainer.load_pretrained_generator(f'./data/checkpoints/srresnet_e{10}.pth.tar')
    srgan_trainer.train(epochs=5)


def train_model_with_adjusted_learning_rate(srgan_trainer, new_learning_rate):
    srgan_trainer.load(f'./data/checkpoints/srgan_e{5}.pth.tar')
    srgan_trainer.change_learning_rate(new_learning_rate=new_learning_rate)
    srgan_trainer.train(epochs=10)


if __name__ == '__main__':
    main()
