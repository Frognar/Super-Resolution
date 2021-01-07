import argparse

import torch
from PIL import Image
from torchvision.transforms import Normalize, ToTensor, ToPILImage

from nn.model import DenseGenerator, Generator
from utils import str2bool


def main():
    args = parse_args()
    model = args.model
    weights = args.weights
    path = args.path
    cuda = args.cuda
    out = args.out

    img = open_image(path)
    img = prepare_image(img)

    if cuda:
        img = process_with_cuda(model, weights, img)
    else:
        img = process_with_cpu(model, weights, img)

    if out:
        save_image(img, out)
    else:
        save_image(img, prepare_new_path(path))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to LR image", type=str)
    parser.add_argument("model", help="SRGAN or ESRGAN", type=str.lower,
                        choices=['srgan', 'esrgan'])
    parser.add_argument("-w", "--weights", help="path to saved model", type=str)
    parser.add_argument("-c", "--cuda", help="Use CUDA? [True]", type=str2bool,
                        default=True)
    parser.add_argument("-o", "--out", help="Path to save SR image", type=str)
    return parser.parse_args()


def process_with_cuda(model, weights, image):
    net = get_model(model, weights).cuda()
    img = make_sr(image.cuda(), net)
    return convert_sr(img.cpu())


def process_with_cpu(model, weights, image):
    net = get_model(model, weights)
    img = make_sr(image, net)
    return convert_sr(img)


def get_model(model, weights):
    if model == 'srgan':
        m = Generator()
    else:
        m = DenseGenerator()

    if weights:
        m.load_state_dict(torch.load(weights)['generator'])

    return m


def open_image(path):
    return Image.open(path).convert('RGB')


def prepare_image(image):
    normalize = Normalize([0.4787, 0.4470, 0.3931], [0.0301, 0.0310, 0.0261])
    return normalize(ToTensor()(image)).unsqueeze(0)


def make_sr(image, model):
    return model(image)


def convert_sr(image):
    return ToPILImage()(image.squeeze(0))


def prepare_new_path(path):
    path = path.split('.')
    path[-1] = '_SR.' + path[-1]
    return ".".join(path[:-1]) + path[-1]


def save_image(image, path):
    image.save(path, quality=100, subsampling=0)


if __name__ == '__main__':
    main()
