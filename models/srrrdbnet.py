from models.srresnet import SRResNet
from models.models import RRDBGenerator


class SRRRDBNet(SRResNet):
    def _initialize_generator(self):
        self._generator = RRDBGenerator().cuda().eval()
