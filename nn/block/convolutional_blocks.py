import torch.nn as nn


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size=3,
                 negative_slope=0.2):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=negative_slope)
        )

    def forward(self, input_data):
        return self.block(input_data)
