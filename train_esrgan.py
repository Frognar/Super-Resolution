from trainers import ESRGANLoggerTrainer

esrgan_params = dict()
esrgan_params['num_workers'] = 4
esrgan_params['batch_size'] = 16
esrgan_params['learning_rate'] = 1e-4
esrgan_params['new_learning_rate'] = 1e-5
esrgan_params['images_list'] = './data/train.json'
esrgan_params['crop_size'] = 128
esrgan_params['print_frequency'] = 3

if __name__ == '__main__':
    esrgan_trainer = ESRGANLoggerTrainer(train_params=esrgan_params)
    esrgan_trainer.load_pretrained_generator(f'./data/checkpoints/srrrdbnet_e{10}.pth.tar')
    esrgan_trainer.train(epochs=5)

    esrgan_trainer.load(f'./data/checkpoints/esrgan_e{5}.pth.tar')
    esrgan_trainer.change_learning_rate(new_learning_rate=esrgan_params['new_learning_rate'])
    esrgan_trainer.train(epochs=10)
