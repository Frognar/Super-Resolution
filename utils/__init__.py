from utils.image_list import ImageListCreator
from utils.loss import ESRGANPerceptualLoss, SRGANPerceptualLoss


def get_gan_training_params(crop_size=96):
    params = get_training_params(crop_size)
    params['new_learning_rate'] = 1e-5
    return params


def get_training_params(crop_size=96):
    return {'num_workers': 4,
            'batch_size': 16,
            'learning_rate': 1e-4,
            'images_list': './data/train.json',
            'crop_size': crop_size,
            'print_frequency': 508}
