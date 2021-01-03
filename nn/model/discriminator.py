import torch.nn as nn

from nn.block.convolutional_blocks import ConvolutionalBlock


class Discriminator(nn.Module):
    def __init__(self, pool=14):
        super().__init__()

        self.convolutional_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            ConvolutionalBlock(in_channels=64, out_channels=64, stride=2),
            ConvolutionalBlock(in_channels=64, out_channels=128, stride=1),
            ConvolutionalBlock(in_channels=128, out_channels=128, stride=2),
            ConvolutionalBlock(in_channels=128, out_channels=256, stride=1),
            ConvolutionalBlock(in_channels=256, out_channels=256, stride=2),
            ConvolutionalBlock(in_channels=256, out_channels=512, stride=1),
            ConvolutionalBlock(in_channels=512, out_channels=512, stride=2)
        )

        self.adaptive_avg_pool_2d = nn.AdaptiveAvgPool2d((pool, pool))

        self.linear_block = nn.Sequential(
            nn.Linear(in_features=512 * pool * pool, out_features=1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=1024, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        batch_size = input_data.size(0)
        output_data = self.convolutional_block(input_data)
        output_data = self.adaptive_avg_pool_2d(output_data)
        return self.linear_block(output_data.view(batch_size, -1))
