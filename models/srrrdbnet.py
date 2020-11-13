from models import RRDBGenerator, SRResNet


class SRRRDBNet(SRResNet):
    def _initialize_generator(self):
        self._generator = RRDBGenerator().cuda().eval()
