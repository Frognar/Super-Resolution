from PIL import Image
from torchvision.transforms import RandomCrop, Resize, ToPILImage, ToTensor


class Converter:
    def __init__(self, crop_size, upscale_factor):
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        self.lr_size = self.crop_size // self.upscale_factor

    def transform(self, image):
        hr_image = self.random_crop_image(image, self.crop_size)
        lr_image = self.resize_image(hr_image, self.lr_size)
        hr_image = self.convert_pil_image_to_tensor_neg_1_1(hr_image)
        lr_image = self.convert_pil_image_to_tensor_0_1(lr_image)
        return hr_image, lr_image

    @staticmethod
    def convert_pil_image_to_tensor_0_1(image):
        return ToTensor()(image)

    @staticmethod
    def convert_pil_image_to_tensor_neg_1_1(image):
        tensor = Converter.convert_pil_image_to_tensor_0_1(image)
        return tensor * 2. - 1.

    @staticmethod
    def convert_tensor_neg_1_1_to_pil_image(tensor):
        tensor = (tensor + 1.) / 2.
        return Converter.convert_tensor_0_1_to_pil_image(tensor)

    @staticmethod
    def convert_tensor_0_1_to_pil_image(tensor):
        return ToPILImage()(tensor)

    @staticmethod
    def random_crop_image(image, crop_size):
        return RandomCrop(crop_size)(image)

    @staticmethod
    def resize_image(image, size):
        return Resize(size, Image.BICUBIC)(image)
