from trainers import ESRGANLoggerTrainer
from utils import get_gan_training_params


def main():
    esrgan_params = get_gan_training_params(crop_size=128)
    esrgan_trainer = ESRGANLoggerTrainer(esrgan_params)
    train_pretrained_model(esrgan_trainer)
    train_model_with_adjusted_learning_rate(esrgan_trainer, new_learning_rate=esrgan_params['new_learning_rate'])


def train_pretrained_model(esrgan_trainer):
    esrgan_trainer.load_pretrained_generator(f'./data/checkpoints/srrrdbnet_e{10}.pth.tar')
    esrgan_trainer.train(epochs=5)


def train_model_with_adjusted_learning_rate(esrgan_trainer, new_learning_rate):
    esrgan_trainer.load(f'./data/checkpoints/esrgan_e{5}.pth.tar')
    esrgan_trainer.change_learning_rate(new_learning_rate=new_learning_rate)
    esrgan_trainer.train(epochs=10)


if __name__ == '__main__':
    main()
