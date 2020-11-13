import torch
from torchvision.transforms import ToPILImage, ToTensor


class SRBaseNet:
    def __init__(self):
        self._initialize_models()

    def _initialize_models(self):
        raise NotImplemented

    def load(self, checkpoint):
        checkpoint = torch.load(checkpoint)
        self._load_models(checkpoint)

    def _load_models(self, checkpoint):
        raise NotImplemented

    def generate(self, lr_image):
        with torch.no_grad():
            lr_image = self._convert_image_to_tensor(lr_image)
            sr_image = self._generate_sr_image(lr_image)
            return self._convert_tensor_to_image(sr_image)

    def _generate_sr_image(self, lr_image):
        raise NotImplemented

    @classmethod
    def _convert_image_to_tensor(cls, image):
        tensor = ToTensor()(image)
        return tensor.unsqueeze(0).cuda()

    @staticmethod
    def _convert_tensor_to_image(tensor):
        tensor = (tensor + 1.) / 2.
        tensor = tensor.squeeze(0).cpu()
        return ToPILImage()(tensor)


class SRBaseGANNet(SRBaseNet):
    def _initialize_models(self):
        self._initialize_generator()
        self._initialize_discriminator()

    def _initialize_generator(self):
        raise NotImplemented

    def _initialize_discriminator(self):
        raise NotImplemented

    def _load_models(self, checkpoint):
        self._load_generator(checkpoint)
        self._load_discriminator(checkpoint)

    def _load_generator(self, checkpoint):
        raise NotImplemented

    def _load_discriminator(self, checkpoint):
        raise NotImplemented

    def _generate_sr_image(self, lr_image):
        raise NotImplemented

    def discriminate(self, image):
        with torch.no_grad():
            image = self._convert_image_to_tensor(image)
            return self._discriminate_image(image)

    def _discriminate_image(self, image):
        raise NotImplemented
