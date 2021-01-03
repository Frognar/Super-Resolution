import torch.nn as nn

from nn.block.residual_blocks import ResidualInResidualDenseBlock
from nn.block.upsample_blocks import UpscaleBlock


class DenseGenerator(nn.Module):

    def __init__(self):
        super().__init__()
        self.convolutional = nn.Conv2d(in_channels=3, out_channels=64,
                                       kernel_size=3, padding=1)

        self.residual_block = nn.Sequential(
            *[ResidualInResidualDenseBlock(channels=64, kernel_size=3, growth=32)
              for _ in range(23)],
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        )

        self.upscale_block = nn.Sequential(
            UpscaleBlock(channels=64, kernel_size=3, upscale_factor=2),
            UpscaleBlock(channels=64, kernel_size=3, upscale_factor=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        output_data = self.convolutional(input_data)
        residual = self.residual_block(output_data)
        return self.upscale_block(output_data + residual)
