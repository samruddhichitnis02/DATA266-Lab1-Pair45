"""CycleGAN models with a larger PatchGAN receptive field.

Generator changes:
- Bilinear Upsample + Conv2d instead of ConvTranspose2d.

Discriminator changes:
- Spectral normalization on convolution layers.
- Approximately 142x142 receptive field instead of the standard 70x70.

The generator still uses the CycleGAN design:
- ReflectionPad2d
- InstanceNorm2d
- Residual blocks
- ReLU activations
- Tanh output
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Two-convolution residual block."""

    def __init__(self, channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ImprovedGenerator(nn.Module):
    """CycleGAN generator using Upsample + Conv2d decoding."""

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        num_residual_blocks: int = 9,
    ):
        super().__init__()

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                input_channels,
                64,
                kernel_size=7,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        channels = 64

        # 256x256 -> 128x128 -> 64x64
        for _ in range(2):
            layers += [
                nn.Conv2d(
                    channels,
                    channels * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(channels * 2),
                nn.ReLU(inplace=True),
            ]
            channels *= 2

        for _ in range(num_residual_blocks):
            layers.append(ResidualBlock(channels))

        # 64x64 -> 128x128 -> 256x256
        for _ in range(2):
            layers += [
                nn.Upsample(
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=False,
                ),
                nn.ReflectionPad2d(1),
                nn.Conv2d(
                    channels,
                    channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.InstanceNorm2d(channels // 2),
                nn.ReLU(inplace=True),
            ]
            channels //= 2

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                channels,
                output_channels,
                kernel_size=7,
                stride=1,
                padding=0,
            ),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class LargePatchGANDiscriminator(nn.Module):
    """Spectral-normalized PatchGAN with approximately 142x142 patches.

    For a 256x256 input, the output is approximately 14x14. Each output
    score sees an approximately 142x142 region of the input image.
    """

    @staticmethod
    def _spectral_conv(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
    ) -> nn.Module:
        return nn.utils.spectral_norm(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        )

    def __init__(self, input_channels: int = 3):
        super().__init__()

        self.model = nn.Sequential(
            # Downsampling: 256 -> 128 -> 64 -> 32 -> 16
            self._spectral_conv(
                input_channels,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),

            self._spectral_conv(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            self._spectral_conv(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Additional downsampling increases the receptive field.
            self._spectral_conv(
                256,
                512,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Two stride-1 layers complete the approximately 142x142 field.
            self._spectral_conv(
                512,
                512,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            self._spectral_conv(
                512,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


__all__ = [
    "ResidualBlock",
    "ImprovedGenerator",
    "LargePatchGANDiscriminator",
    "count_parameters",
]
