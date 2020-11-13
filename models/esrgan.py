from models import Discriminator, RRDBGenerator, SRGAN


class ESRGAN(SRGAN):
    def _initialize_models(self):
        self._generator = RRDBGenerator().cuda().eval()
        self._discriminator = Discriminator(is_pooling_needed=True, pool_size=8).cuda().eval()
