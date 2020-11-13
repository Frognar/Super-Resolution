from models.models import Discriminator, RRDBGenerator
from models.srgan import SRGAN


class ESRGAN(SRGAN):
    def _initialize_generator(self):
        self._generator = RRDBGenerator().cuda().eval()

    def _initialize_discriminator(self):
        self._discriminator = Discriminator(is_pooling_needed=True, pool_size=8).cuda().eval()
