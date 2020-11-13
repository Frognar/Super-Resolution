from trainers import SRGANLoggerTrainer

srgan_params = dict()
srgan_params['num_workers'] = 4
srgan_params['batch_size'] = 16
srgan_params['learning_rate'] = 1e-4
srgan_params['new_learning_rate'] = 1e-5
srgan_params['images_list'] = './data/train.json'
srgan_params['crop_size'] = 96
srgan_params['print_frequency'] = 508

if __name__ == '__main__':
    srgan_trainer = SRGANLoggerTrainer(train_params=srgan_params)
    srgan_trainer.load_pretrained_generator(f'./data/checkpoints/srresnet_e{10}.pth.tar')
    srgan_trainer.train(epochs=5)

    srgan_trainer.load(f'./data/checkpoints/srgan_e{5}.pth.tar')
    srgan_trainer.change_learning_rate(new_learning_rate=srgan_params['new_learning_rate'])
    srgan_trainer.train(epochs=10)
