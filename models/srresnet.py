from models.models import Generator
from models.srbase import SRBaseNet


class SRResNet(SRBaseNet):
    def _initialize_models(self):
        self._generator = Generator().cuda().eval()

    def _load_models(self, checkpoint):
        self._generator.load_state_dict(checkpoint['generator'])

    def _generate_sr_image(self, lr_image):
        return self._generator(lr_image)
