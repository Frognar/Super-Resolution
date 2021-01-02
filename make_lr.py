import argparse
from PIL import Image


def main():
    args = parse_args()
    path = args.path
    factor = args.factor
    out = args.out

    img = open_image(path)
    img = make_lr(img, factor)
    if out:
        save_image(img, out)
    else:
        save_image(img, prepare_new_path(path))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to HR image", type=str)
    parser.add_argument("factor", help="Scale down factor [int]", type=int)
    parser.add_argument("-o", "--out", help="Path to save LR image", type=str)
    return parser.parse_args()


def open_image(path):
    return Image.open(path).convert('RGB')


def make_lr(image, factor):
    return image.resize((image.width // factor, image.height // factor), Image.BICUBIC)


def prepare_new_path(path):
    path = path.split('.')
    path[-1] = '_LR.' + path[-1]
    return ".".join(path[:-1]) + path[-1]


def save_image(image, path):
    image.save(path, quality=100, subsampling=0)


if __name__ == '__main__':
    main()
