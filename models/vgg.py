import torch
import torchvision


class TruncatedVGG19(torch.nn.Module):
    def __init__(self):
        super(TruncatedVGG19, self).__init__()
        vgg19 = torchvision.models.vgg19(pretrained=True)
        self.net = torch.nn.Sequential(*list(vgg19.features.children())[:36]).eval()

    def forward(self, input_data):
        return self.net(input_data)


class TruncatedVGG19BeforeActivation(torch.nn.Module):
    def __init__(self):
        super(TruncatedVGG19BeforeActivation, self).__init__()
        vgg19 = torchvision.models.vgg19(pretrained=True)
        self.net = torch.nn.Sequential(*list(vgg19.features.children())[:35]).eval()

    def forward(self, input_data):
        return self.net(input_data)
