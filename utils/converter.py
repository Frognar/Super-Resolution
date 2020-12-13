import PIL
from torchvision.transforms import Normalize, RandomCrop, Resize, ToTensor


class Converter:
    def __init__(self, crop_size=224, upscale_factor=4, mean=None, std=None):
        if mean is None:
            mean = [0.4787, 0.4470, 0.3931]

        if std is None:
            std = [0.0301, 0.0310, 0.0261]

        self.random_crop = RandomCrop(crop_size)
        self.resize = Resize(crop_size // upscale_factor, PIL.Image.BICUBIC)
        self.convert = ToTensor()
        self.normalize = Normalize(mean, std)

    def transform(self, image):
        hr_image = self.random_crop(image)
        lr_image = self.resize(hr_image)
        return self.convert(hr_image), self.normalize(self.convert(lr_image))
