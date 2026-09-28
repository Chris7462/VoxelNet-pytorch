import torch.nn as nn
from torch import Tensor


def conv_bn_relu(in_channels: int, out_channels: int, stride: int = 1) -> list[nn.Module]:
    """Conv2d(3x3) → BN → ReLU."""
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]


def make_block(in_channels: int, out_channels: int, num_layers: int) -> nn.Sequential:
    """One stride-2 conv followed by `num_layers` stride-1 convs, all Conv → BN → ReLU."""
    layers = conv_bn_relu(in_channels, out_channels, stride=2)
    for _ in range(num_layers):
        layers += conv_bn_relu(out_channels, out_channels)
    return nn.Sequential(*layers)


class RPNBackbone(nn.Module):
    """
    Convolutional blocks of the VoxelNet Region Proposal Network.

    Architecture (every conv is followed by BN → ReLU):
        Block 1: Conv(C → 128, s2) + 3 × Conv(128 → 128)   → (B, 128, H/2, W/2)
        Block 2: Conv(128 → 128, s2) + 5 × Conv(128 → 128) → (B, 128, H/4, W/4)
        Block 3: Conv(128 → 256, s2) + 5 × Conv(256 → 256) → (B, 256, H/8, W/8)
    """

    def __init__(self, in_channels: int = 128) -> None:
        super().__init__()

        self.block_1 = make_block(in_channels, 128, num_layers=3)
        self.block_2 = make_block(128, 128, num_layers=5)
        self.block_3 = make_block(128, 256, num_layers=5)

        self.out_channels = (128, 128, 256)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x: BEV feature map (B, C, H, W); H and W must be divisible by 8

        Returns:
            Feature maps of the three blocks at strides 2, 4 and 8
        """
        x1 = self.block_1(x)
        x2 = self.block_2(x1)
        x3 = self.block_3(x2)
        return x1, x2, x3
