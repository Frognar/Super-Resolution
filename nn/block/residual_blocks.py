import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels=64, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ),
            nn.BatchNorm2d(num_features=channels),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ),
            nn.BatchNorm2d(num_features=channels)
        )

    def forward(self, input_data):
        return input_data + self.block(input_data)


class ResidualDenseBlock(nn.Module):
    def __init__(self, channels=64, kernel_size=3, growth=32):
        super().__init__()
        self.conv2d_1 = self.conv2d(channels, growth, kernel_size, growth, 0)
        self.conv2d_2 = self.conv2d(channels, growth, kernel_size, growth, 1)
        self.conv2d_3 = self.conv2d(channels, growth, kernel_size, growth, 2)
        self.conv2d_4 = self.conv2d(channels, growth, kernel_size, growth, 3)
        self.conv2d_5 = self.conv2d(channels, channels, kernel_size, growth, 4)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    @staticmethod
    def conv2d(in_channels, out_channels, kernel_size, growth, factor):
        return nn.Conv2d(
            in_channels=in_channels + factor * growth,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

    def forward(self, input_data):
        x1 = self.relu(self.conv2d_1(input_data))
        x2 = self.relu(self.conv2d_2(torch.cat((input_data, x1), 1)))
        x3 = self.relu(self.conv2d_3(torch.cat((input_data, x1, x2), 1)))
        x4 = self.relu(self.conv2d_4(torch.cat((input_data, x1, x2, x3), 1)))
        x5 = self.conv2d_5(torch.cat((input_data, x1, x2, x3, x4), 1))
        return input_data + x5 * 0.2


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, channels=64, kernel_size=3, growth=32):
        super().__init__()
        self.block = nn.Sequential(
            ResidualDenseBlock(channels, kernel_size, growth),
            ResidualDenseBlock(channels, kernel_size, growth),
            ResidualDenseBlock(channels, kernel_size, growth)
        )

    def forward(self, input_data):
        return input_data + self.block(input_data) * 0.2
