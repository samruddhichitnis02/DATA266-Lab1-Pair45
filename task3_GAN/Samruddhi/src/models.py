import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    - Residual block used inside the CycleGAN generator.
    - The generator’s task is to translate an image from one domain into the style of another domain.
    - There are 2 generators G_AB: ordinary photo → Monet-style painting and G_BA: Monet painting → ordinary photo
    - Each generator contains 9 residual blocks.
    - The complete CycleGAN has:
        - 9 residual blocks in G_AB
        - 9 residual blocks in G_BA
    - The discriminators do not use residual blocks.
    - The residual block helps by adding the learned change to the original features
    - x : original image features or feature map of the original image
    - self.block(x) : Applies convolution, normalization, and activation layers to the feature map
    - returns the original image feature map + applied changes
    """

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),

            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=0
            ),

            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),

            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=0
            ),

            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        # Skip connection
        return x + self.block(x)


class Generator(nn.Module):
    """
    CycleGAN generator.

    For 256x256 images, the paper uses:
    c7s1-64, d128, d256,
    nine residual blocks,
    u128, u64, c7s1-3

    - Input image → extract features → learn style changes → reconstruct output image
    - For example : Photo → Monet-style photo
    """

    def __init__(
        self,
        input_channels=3,
        output_channels=3,
        num_residual_blocks=9
    ):
        super().__init__()

        layers = [
            # Initial feature extraction
            nn.ReflectionPad2d(3),

            nn.Conv2d(
                input_channels,
                64,
                kernel_size=7,
                stride=1,
                padding=0
            ),

            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling: 256x256 -> 128x128 -> 64x64
        channels = 64

        for _ in range(2):
            layers += [
                nn.Conv2d(
                    channels,
                    channels * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),

                nn.InstanceNorm2d(channels * 2),
                nn.ReLU(inplace=True)
            ]

            channels *= 2

        # Nine residual blocks for 256x256 images
        for _ in range(num_residual_blocks):
            layers.append(ResidualBlock(channels))

        # Upsampling: 64x64 -> 128x128 -> 256x256
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(
                    channels,
                    channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),

                nn.InstanceNorm2d(channels // 2),
                nn.ReLU(inplace=True)
            ]

            channels //= 2

        # Convert features back to an RGB image
        layers += [
            nn.ReflectionPad2d(3),

            nn.Conv2d(
                channels,
                output_channels,
                kernel_size=7,
                stride=1,
                padding=0
            ),

            # Output range: [-1, 1]
            nn.Tanh()
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PatchGANDiscriminator(nn.Module):
    """
    - 70x70 PatchGAN discriminator used in CycleGAN.
    - Its job is to look at an image and decide whether its local regions look real or fake.
    - Photo discriminator: real photo or generated photo?
    - Monet discriminator: real Monet painting or generated Monet painting?
    """

    def __init__(self, input_channels=3):
        super().__init__()

        self.model = nn.Sequential(
            # First layer does not use InstanceNorm
            nn.Conv2d(
                input_channels,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                256,
                512,
                kernel_size=4,
                stride=1,
                padding=1
            ),

            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # One score for each local image patch
            nn.Conv2d(
                512,
                1,
                kernel_size=4,
                stride=1,
                padding=1
            )
        )

    def forward(self, x):
        return self.model(x)