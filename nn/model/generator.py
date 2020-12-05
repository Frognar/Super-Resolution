import torch.nn as nn

from nn.block.residual_blocks import ResidualBlock
from nn.block.upsample_blocks import PixelShuffleUpscaleBlock


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolutional_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=9,
                padding=4),
            nn.PReLU()
        )

        self.residual_block = nn.Sequential(
            *[ResidualBlock() for _ in range(16)]
        )
        self.convolutional_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(num_features=64)
        )

        self.upscale_block = nn.Sequential(
            PixelShuffleUpscaleBlock(),
            PixelShuffleUpscaleBlock(),
            nn.Conv2d(
                in_channels=64,
                out_channels=3,
                kernel_size=9,
                padding=4
            ),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        output_data = self.convolutional_block_1(input_data)
        residual = output_data
        output_data = self.residual_block(output_data)
        output_data = self.convolutional_block_2(output_data)
        return self.upscale_block(output_data + residual)
