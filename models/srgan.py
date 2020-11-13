from models.models import Discriminator, Generator
from models.srbase import SRBaseGANNet


class SRGAN(SRBaseGANNet):
    def _initialize_generator(self):
        self._generator = Generator().cuda().eval()

    def _initialize_discriminator(self):
        self._discriminator = Discriminator(is_pooling_needed=True).cuda().eval()

    def _load_generator(self, checkpoint):
        self._generator.load_state_dict(checkpoint['generator'])

    def _load_discriminator(self, checkpoint):
        self._discriminator.load_state_dict(checkpoint['discriminator'])

    def _generate_sr_image(self, lr_image):
        return self._generator(lr_image)

    def _discriminate_image(self, image):
        return self._discriminator(image)
