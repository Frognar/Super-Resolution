import torch.nn as nn
from torch.nn.functional import interpolate


class PixelShuffleUpscaleBlock(nn.Module):
    def __init__(self, in_channels=64, kernel_size=3, upscale_factor=2):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=in_channels * (upscale_factor ** 2),
                      kernel_size=kernel_size, padding=kernel_size // 2),
            nn.PixelShuffle(upscale_factor=upscale_factor),
            nn.PReLU()
        )

    def forward(self, input_data):
        return self.block(input_data)


class UpscaleBlock(nn.Module):
    def __init__(self, channels=64, kernel_size=3, upscale_factor=2):
        super().__init__()
        self.scale_factor = upscale_factor
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, input_data):
        return self.block(self.upscale(input_data))

    def upscale(self, data):
        return interpolate(data, scale_factor=self.scale_factor, mode='nearest')
