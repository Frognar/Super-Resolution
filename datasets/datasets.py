import json

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import RandomCrop, Resize, ToTensor


class TrainDataset(Dataset):
    def __init__(self, image_list_path, crop_size, upscale_factor):
        super().__init__()
        self._crop_size = crop_size
        self._lr_size = self._crop_size // upscale_factor
        self._image_list = self._get_image_list(image_list_path)

    @staticmethod
    def _get_image_list(image_list_path):
        with open(image_list_path, mode='r') as json_images:
            return json.load(json_images)

    def __getitem__(self, index):
        image = self._open_image(index)
        lr_image, hr_image = self._transform_images(image)
        return lr_image, hr_image

    def _open_image(self, index):
        image_path = self._image_list[index]
        image = Image.open(image_path)
        return image.convert(mode='RGB')

    def _transform_images(self, image):
        hr_image = self._crop_image(image)
        lr_image = self._resize_image(hr_image)
        hr_image = self._transform_image_to_neg_1_1_range_tensor(hr_image)
        lr_image = self._transform_image_to_0_1_range_tensor(lr_image)
        return hr_image, lr_image

    def _crop_image(self, image):
        return RandomCrop(self._crop_size)(image)

    def _resize_image(self, image):
        return Resize(self._lr_size, Image.BICUBIC)(image)

    @classmethod
    def _transform_image_to_neg_1_1_range_tensor(cls, image):
        return cls._transform_image_to_0_1_range_tensor(image) * 2. - 1.

    @staticmethod
    def _transform_image_to_0_1_range_tensor(image):
        return ToTensor()(image)

    def __len__(self):
        return len(self._image_list)
