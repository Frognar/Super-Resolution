import torch
from torchvision.transforms import ToPILImage, ToTensor

from models.models import Generator


class SRResNet:
    def __init__(self):
        self.__initialize_generator()

    def __initialize_generator(self):
        self.__generator = Generator().cuda().eval()

    def load(self, checkpoint):
        checkpoint = torch.load(checkpoint)
        self.__generator.load_state_dict(checkpoint['generator'])

    def generate(self, lr_image):
        with torch.no_grad():
            lr_image = SRResNet._convert_image_to_tensor(lr_image)
            sr_image = self.__generator(lr_image)
            return SRResNet._convert_tensor_to_image(sr_image)

    @classmethod
    def _convert_image_to_tensor(cls, image):
        tensor = ToTensor()(image)
        return tensor.unsqueeze(0).cuda()

    @staticmethod
    def _convert_tensor_to_image(tensor):
        tensor = (tensor + 1.) / 2.
        tensor = tensor.squeeze(0).cpu()
        return ToPILImage()(tensor)