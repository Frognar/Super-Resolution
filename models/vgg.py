import torch
import torchvision


class TruncatedVGG19(torch.nn.Module):
    def __init__(self, with_activation_layer=True):
        super(TruncatedVGG19, self).__init__()
        vgg19 = torchvision.models.vgg19(pretrained=True)

        layers = 36 if with_activation_layer else 35
        self.net = torch.nn.Sequential(
            *list(vgg19.features.children())[:layers]
        ).eval()

    def forward(self, input_data):
        return self.net(input_data)
