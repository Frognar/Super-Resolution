import torch


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels=64, kernel_size=3):
        super(ResidualBlock, self).__init__()

        self.__convolutional_1 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                                 padding=kernel_size // 2)
        self.__batch_normalization_1 = torch.nn.BatchNorm2d(num_features=channels)
        self.__prelu = torch.nn.PReLU()
        self.__convolutional_2 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                                 padding=kernel_size // 2)
        self.__batch_normalization_2 = torch.nn.BatchNorm2d(num_features=channels)

    def forward(self, input_data):
        output_data = self.__convolutional_1(input_data)
        output_data = self.__batch_normalization_1(output_data)
        output_data = self.__prelu(output_data)
        output_data = self.__convolutional_2(output_data)
        output_data = self.__batch_normalization_2(output_data)
        return output_data + input_data


class ResidualDenseBlock(torch.nn.Module):
    def __init__(self, channel=64, growth_channel=32, bias=True):
        super().__init__()
        self.__convolution_1 = torch.nn.Conv2d(in_channels=channel, out_channels=growth_channel,
                                               kernel_size=3, padding=1, bias=bias)
        self.__convolution_2 = torch.nn.Conv2d(in_channels=channel, out_channels=channel + growth_channel,
                                               kernel_size=3, padding=1, bias=bias)
        self.__convolution_3 = torch.nn.Conv2d(in_channels=channel, out_channels=channel + 2 * growth_channel,
                                               kernel_size=3, padding=1, bias=bias)
        self.__convolution_4 = torch.nn.Conv2d(in_channels=channel, out_channels=channel + 3 * growth_channel,
                                               kernel_size=3, padding=1, bias=bias)
        self.__convolution_5 = torch.nn.Conv2d(in_channels=channel, out_channels=channel + 4 * growth_channel,
                                               kernel_size=3, padding=1, bias=bias)
        self.__leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, input_data):
        out_1 = self.__convolution_1(input_data)
        out_1 = self.__leaky_relu(out_1)
        out_2 = self.__convolution_2(torch.cat((input_data, out_1), 1))
        out_2 = self.__leaky_relu(out_2)
        out_3 = self.__convolution_3(torch.cat((input_data, out_1, out_2), 1))
        out_3 = self.__leaky_relu(out_3)
        out_4 = self.__convolution_4(torch.cat((input_data, out_1, out_2, out_3), 1))
        out_4 = self.__leaky_relu(out_4)
        out_5 = self.__convolution_5(torch.cat((input_data, out_1, out_2, out_3, out_4), 1))
        return input_data + out_5 * 0.2


class ResidualInResidualDenseBlock(torch.nn.Module):
    def __init__(self, channel, growth_channel=32):
        super().__init__()
        self.__residual_dense_block_1 = ResidualDenseBlock(channel, growth_channel)
        self.__residual_dense_block_2 = ResidualDenseBlock(channel, growth_channel)
        self.__residual_dense_block_3 = ResidualDenseBlock(channel, growth_channel)

    def forward(self, input_data):
        output_data = self.__residual_dense_block_1(input_data)
        output_data = self.__residual_dense_block_2(output_data)
        output_data = self.__residual_dense_block_3(output_data)
        return input_data + output_data * 0.2


class UpsampleBlock(torch.nn.Module):
    def __init__(self, in_channels=64, kernel_size=3, upscale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.__convolutional = torch.nn.Conv2d(in_channels=in_channels,
                                               out_channels=in_channels * (upscale_factor ** 2),
                                               kernel_size=kernel_size, padding=kernel_size // 2)
        self.__pixel_shuffle = torch.nn.PixelShuffle(upscale_factor=upscale_factor)
        self.__prelu = torch.nn.PReLU()

    def forward(self, input_data):
        output_data = self.__convolutional(input_data)
        output_data = self.__pixel_shuffle(output_data)
        output_data = self.__prelu(output_data)
        return output_data


class ConvolutionalBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, negative_slope=0.2):
        super(ConvolutionalBlock, self).__init__()

        self.__convolutional = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                               kernel_size=kernel_size,
                                               stride=stride, padding=kernel_size // 2)
        self.__batch_normalization = torch.nn.BatchNorm2d(num_features=out_channels)
        self.__leaky_relu = torch.nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, input_data):
        output_data = self.__convolutional(input_data)
        output_data = self.__batch_normalization(output_data)
        output_data = self.__leaky_relu(output_data)
        return output_data
