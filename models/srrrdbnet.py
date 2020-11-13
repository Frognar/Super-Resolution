from models.models import RRDBGenerator
from models.srresnet import SRResNet


class SRRRDBNet(SRResNet):
    def _initialize_models(self):
        self._generator = RRDBGenerator().cuda().eval()
