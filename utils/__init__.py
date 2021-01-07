import argparse

from utils.image_list_creator import ImageListCreator
from utils.converter import Converter


def parse_training_args(model_name):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", "-e", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--load", "-l", type=str,
                        help="Path to .pth.tar file with trained model")
    if "SRGAN" in model_name:
        parser.add_argument("--init", "-i", type=str,
                            help="Path to .pth.tar file with pretrained model")
    parser.add_argument("--out", "-o", type=str,
                        default=f"data/checkpoints/{model_name}.pth.tar",
                        help=f"Path to save trained model"
                             f" [data/checkpoints/{model_name}.pth.tar]")
    parser.add_argument("--cuda", "-c", type=str2bool, default=True,
                        help="Use CUDA? [True]")
    return parser.parse_args()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ['yes', 'true', 't', 'y', '1']:
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
