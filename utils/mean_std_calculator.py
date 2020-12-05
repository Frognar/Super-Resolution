import torch
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, Resize, ToTensor
from tqdm import tqdm

from dataset import ImageNetDataset


class DummyConverter:
    @staticmethod
    def transform(img):
        return ToTensor()(CenterCrop(224)(Resize(256)(img)))


def calculate_mean_std():
    dataset = ImageNetDataset('data/train.json', DummyConverter())
    data_loader = DataLoader(dataset, batch_size=512, shuffle=True)

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for img in tqdm(data_loader):
        channels_sum += torch.mean(img, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(img ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean) ** 2
    return mean, std


if __name__ == '__main__':
    print(*calculate_mean_std())
