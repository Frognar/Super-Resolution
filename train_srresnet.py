from trainers.srresnet_trainer import SRResNetLoggerTrainer

srresnet_params = dict()
srresnet_params['num_workers'] = 4
srresnet_params['batch_size'] = 16
srresnet_params['learning_rate'] = 1e-4
srresnet_params['images_list'] = './data/train.json'
srresnet_params['crop_size'] = 96
srresnet_params['print_frequency'] = 500

if __name__ == '__main__':
    srresnet_trainer = SRResNetLoggerTrainer(train_params=srresnet_params)
    srresnet_trainer.train(epochs=10)
