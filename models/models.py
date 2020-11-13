import torch

from models.blocks import ConvolutionalBlock, ResidualBlock, ResidualInResidualDenseBlock, UpsampleBlock


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.__convolutional_block_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4),
            torch.nn.PReLU())

        self.__residual_block = torch.nn.Sequential(*[ResidualBlock(channels=64, kernel_size=3) for _ in range(16)])

        self.__convolutional_block_2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(num_features=64))

        self.__upsample_block = torch.nn.Sequential(
            UpsampleBlock(in_channels=64, kernel_size=3, upscale_factor=2),
            UpsampleBlock(in_channels=64, kernel_size=3, upscale_factor=2),
            torch.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, padding=4),
            torch.nn.Tanh())

    def forward(self, input_data):
        output_data = self.__convolutional_block_1(input_data)
        residual = output_data
        output_data = self.__residual_block(output_data)
        output_data = self.__convolutional_block_2(output_data)
        output_data = output_data + residual
        return self.__upsample_block(output_data)


class Discriminator(torch.nn.Module):
    def __init__(self, is_pooling_needed=False, pool_size=6):
        super(Discriminator, self).__init__()
        self.__convolutional_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(negative_slope=0.2),
            ConvolutionalBlock(in_channels=64, out_channels=64, kernel_size=3, stride=2),
            ConvolutionalBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            ConvolutionalBlock(in_channels=128, out_channels=128, kernel_size=3, stride=2),
            ConvolutionalBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            ConvolutionalBlock(in_channels=256, out_channels=256, kernel_size=3, stride=2),
            ConvolutionalBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1),
            ConvolutionalBlock(in_channels=512, out_channels=512, kernel_size=3, stride=2))

        self.__linear_block = torch.nn.Sequential(
            torch.nn.Linear(in_features=512 * pool_size * pool_size, out_features=1024),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Linear(in_features=1024, out_features=1),
            torch.nn.Sigmoid())

        self.__is_pooling_needed = is_pooling_needed
        self.__pool_size = pool_size

    def forward(self, input_data):
        batch_size = input_data.size(0)
        output_data = self.__convolutional_block(input_data)

        if self.__is_pooling_needed:
            output_data = self.__normalize_tensor_size(output_data)

        return self.__linear_block(output_data.view(batch_size, -1))

    def __normalize_tensor_size(self, tensor):
        return torch.nn.AdaptiveAvgPool2d((self.__pool_size, self.__pool_size))(tensor)


class RRDBGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.__convolutional_1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=True)

        self.__residual_block = torch.nn.Sequential(
            *[ResidualInResidualDenseBlock(channel=64, growth_channel=32) for _ in range(23)],
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True))

        self.__upscale_block_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.__upscale_block_2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.__convolutional_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1, bias=True),
            torch.nn.Tanh())

    def forward(self, input_data):
        output_data = self.__convolutional_1(input_data)
        residual = self.__residual_block(output_data)
        output_data = output_data + residual

        def upscale(data): return torch.nn.functional.interpolate(data, scale_factor=2, mode='nearest')

        output_data = upscale(output_data)
        output_data = self.__upscale_block_1(output_data)

        output_data = upscale(output_data)
        output_data = self.__upscale_block_2(output_data)

        output_data = self.__convolutional_block(output_data)

        return output_data
