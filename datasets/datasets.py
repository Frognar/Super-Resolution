import json

from PIL import Image
from torch.utils.data.dataset import Dataset


class TrainDataset(Dataset):
    def __init__(self, image_list_path, converter):
        super().__init__()
        self.image_list = self.get_image_list(image_list_path)
        self.converter = converter

    @staticmethod
    def get_image_list(image_list_path):
        with open(image_list_path, mode='r') as json_images:
            return json.load(json_images)

    def __getitem__(self, index):
        return self.converter.transform(self.open(index))

    def open(self, index):
        image = Image.open(self.image_list[index])
        return image.convert(mode='RGB')

    def __len__(self):
        return len(self.image_list)
