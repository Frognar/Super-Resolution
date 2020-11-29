from torch import cat
from torch.nn import BatchNorm2d, Conv2d, LeakyReLU, Module, PReLU, \
    PixelShuffle, Sequential


class ResidualBlock(Module):
    def __init__(self, channels=64, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.residual_block = Sequential(
            Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ),
            BatchNorm2d(num_features=channels),
            PReLU(),
            Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ),
            BatchNorm2d(num_features=channels)
        )

    def forward(self, input_data):
        return input_data + self.residual_block(input_data)


class ResidualDenseBlock(Module):
    def __init__(self, channel=64, growth=32):
        super().__init__()
        self.convolution_1 = self.conv2d(channel, growth, growth, 0)
        self.convolution_2 = self.conv2d(channel, growth, growth, 1)
        self.convolution_3 = self.conv2d(channel, growth, growth, 2)
        self.convolution_4 = self.conv2d(channel, growth, growth, 3)
        self.convolution_5 = self.conv2d(channel, channel, growth, 4)
        self.leaky_relu = LeakyReLU(negative_slope=0.2, inplace=True)

    @staticmethod
    def conv2d(in_channels, out_channels, growth, i):
        return Conv2d(
            in_channels=in_channels + i * growth,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, input_data):
        x1 = self.convolution_1(input_data)
        x1 = self.leaky_relu(x1)
        x2 = self.convolution_2(cat((input_data, x1), 1))
        x2 = self.leaky_relu(x2)
        x3 = self.convolution_3(cat((input_data, x1, x2), 1))
        x3 = self.leaky_relu(x3)
        x4 = self.convolution_4(cat((input_data, x1, x2, x3), 1))
        x4 = self.leaky_relu(x4)
        x5 = self.convolution_5(cat((input_data, x1, x2, x3, x4), 1))
        return input_data + x5 * 0.2


class ResidualInResidualDenseBlock(Module):
    def __init__(self, channel, growth_channel=32):
        super().__init__()
        self.residual_dense_block = Sequential(
            ResidualDenseBlock(channel, growth_channel),
            ResidualDenseBlock(channel, growth_channel),
            ResidualDenseBlock(channel, growth_channel)
        )

    def forward(self, input_data):
        return input_data + self.residual_dense_block(input_data) * 0.2


class UpsampleBlock(Module):
    def __init__(self, in_channels=64, kernel_size=3, upscale_factor=2):
        super(UpsampleBlock, self).__init__()

        self.block = Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=in_channels * (upscale_factor ** 2),
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ),
            PixelShuffle(upscale_factor=upscale_factor),
            PReLU()
        )

    def forward(self, input_data):
        return self.block(input_data)


class ConvolutionalBlock(Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size=3,
                 negative_slope=0.2):
        super(ConvolutionalBlock, self).__init__()

        self.block = Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2
            ),
            BatchNorm2d(num_features=out_channels),
            LeakyReLU(negative_slope=negative_slope)
        )

    def forward(self, input_data):
        return self.block(input_data)
