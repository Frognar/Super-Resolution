from torch.nn import AdaptiveAvgPool2d, BatchNorm2d, Conv2d, LeakyReLU, \
    Linear, Module, PReLU, Sequential, Sigmoid, Tanh
from torch.nn.functional import interpolate

from models.blocks import ConvolutionalBlock, ResidualBlock, \
    ResidualInResidualDenseBlock, UpsampleBlock


class Generator(Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.convolutional_block_1 = Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4),
            PReLU()
        )

        self.residual_block = Sequential(*[ResidualBlock() for _ in range(16)])
        self.convolutional_block_2 = Sequential(
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            BatchNorm2d(num_features=64)
        )

        self.upsample_block = Sequential(
            UpsampleBlock(),
            UpsampleBlock(),
            Conv2d(in_channels=64, out_channels=3, kernel_size=9, padding=4),
            Tanh()
        )

    def forward(self, input_data):
        output_data = self.convolutional_block_1(input_data)
        residual = output_data
        output_data = self.residual_block(output_data)
        output_data = self.convolutional_block_2(output_data)
        output_data = output_data + residual
        return self.upsample_block(output_data)


class Discriminator(Module):
    def __init__(self, is_pooling_needed=False, pool_size=6):
        super(Discriminator, self).__init__()
        self.convolutional_block = Sequential(
            Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            LeakyReLU(negative_slope=0.2),
            ConvolutionalBlock(in_channels=64, out_channels=64, stride=2),
            ConvolutionalBlock(in_channels=64, out_channels=128, stride=1),
            ConvolutionalBlock(in_channels=128, out_channels=128, stride=2),
            ConvolutionalBlock(in_channels=128, out_channels=256, stride=1),
            ConvolutionalBlock(in_channels=256, out_channels=256, stride=2),
            ConvolutionalBlock(in_channels=256, out_channels=512, stride=1),
            ConvolutionalBlock(in_channels=512, out_channels=512, stride=2)
        )

        self.linear_block = Sequential(
            Linear(in_features=512 * pool_size * pool_size, out_features=1024),
            LeakyReLU(negative_slope=0.2),
            Linear(in_features=1024, out_features=1),
            Sigmoid()
        )

        self.is_pooling_needed = is_pooling_needed
        self.pool_size = pool_size

    def forward(self, input_data):
        batch_size = input_data.size(0)
        output_data = self.convolutional_block(input_data)

        if self.is_pooling_needed:
            output_data = self.normalize_tensor_size(output_data)

        return self.linear_block(output_data.view(batch_size, -1))

    def normalize_tensor_size(self, tensor):
        return AdaptiveAvgPool2d((self.pool_size, self.pool_size))(tensor)


class RRDBGenerator(Module):
    def __init__(self):
        super().__init__()
        self.convolutional_1 = Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.residual_block = Sequential(
            *[ResidualInResidualDenseBlock(channel=64, growth_channel=32)
              for _ in range(23)],
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        )

        self.upscale_block_1 = Sequential(
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.upscale_block_2 = Sequential(
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.convolutional_block = Sequential(
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            LeakyReLU(negative_slope=0.2, inplace=True),
            Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1),
            Tanh()
        )

    def forward(self, input_data):
        output_data = self.convolutional_1(input_data)
        residual = self.residual_block(output_data)
        output_data = output_data + residual

        def upscale(data):
            return interpolate(data, scale_factor=2, mode='nearest')

        output_data = self.upscale_block_1(upscale(output_data))
        output_data = self.upscale_block_2(upscale(output_data))
        return self.convolutional_block(output_data)
