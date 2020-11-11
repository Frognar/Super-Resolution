import torch
from torchvision.transforms import ToPILImage, ToTensor

from models.models import Discriminator, Generator


class SRGAN:
    def __init__(self):
        self.__initialize_models()

    def __initialize_models(self):
        self.__generator = Generator().cuda().eval()
        self.__discriminator = Discriminator(is_pooling_needed=True).cuda().eval()

    def load(self, checkpoint):
        checkpoint = torch.load(checkpoint)
        self.__generator.load_state_dict(checkpoint['generator'])
        self.__discriminator.load_state_dict(checkpoint['discriminator'])

    def generate(self, lr_image):
        with torch.no_grad():
            lr_image = SRGAN._convert_image_to_tensor(lr_image)
            sr_image = self.__generator(lr_image)
            return SRGAN._convert_tensor_to_image(sr_image)

    def discriminate(self, image):
        image = SRGAN._convert_image_to_tensor(image)
        return self.__discriminator(image)

    @classmethod
    def _convert_image_to_tensor(cls, image):
        tensor = ToTensor()(image)
        return tensor.unsqueeze(0).cuda()

    @staticmethod
    def _convert_tensor_to_image(tensor):
        tensor = (tensor + 1.) / 2.
        tensor = tensor.squeeze(0).cpu()
        return ToPILImage()(tensor)
