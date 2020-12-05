import json

from PIL import Image
from torch.utils.data.dataset import Dataset


class ImageNetDataset(Dataset):
    def __init__(self, json_path, converter):
        super(ImageNetDataset, self).__init__()
        self.images = ImageNetDataset.load_images(json_path)
        self.converter = converter

    @staticmethod
    def load_images(json_path):
        with open(json_path, mode='r') as json_file:
            images = json.load(json_file)
        return images

    def open_image(self, index):
        return Image.open(self.images[index]).convert(mode='RGB')

    def __getitem__(self, index):
        return self.converter.transform(self.open_image(index))

    def __len__(self):
        return len(self.images)
