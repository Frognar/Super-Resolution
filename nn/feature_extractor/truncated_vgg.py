import torch.nn as nn
import torchvision.models as models


class TruncatedVgg(nn.Module):
    def __init__(self, with_activation_layer=True):
        super().__init__()

        vgg19 = models.vgg19(pretrained=True)
        layers = 36 if with_activation_layer else 35

        self.net = nn.Sequential(*list(vgg19.features.children())[:layers]).eval()

    def forward(self, input_data):
        return self.net(input_data)
