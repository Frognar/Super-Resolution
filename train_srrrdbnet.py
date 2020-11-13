from trainers import SRRRDBNetLoggerTrainer

srrrdbnet_params = dict()
srrrdbnet_params['num_workers'] = 4
srrrdbnet_params['batch_size'] = 16
srrrdbnet_params['learning_rate'] = 1e-4
srrrdbnet_params['images_list'] = './data/train.json'
srrrdbnet_params['crop_size'] = 128
srrrdbnet_params['print_frequency'] = 508

if __name__ == '__main__':
    srrrdbnet_trainer = SRRRDBNetLoggerTrainer(train_params=srrrdbnet_params)
    srrrdbnet_trainer.train(epochs=10)
