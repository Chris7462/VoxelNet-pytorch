import torch.nn as nn
from torch import Tensor


def _conv3d_out_depth(depth: int, stride: int, padding: int, kernel_size: int = 3) -> int:
    return (depth + 2 * padding - kernel_size) // stride + 1


class ConvMiddleLayer(nn.Module):
    """
    Convolutional middle layers: 3D convolutions over the dense voxel grid.

    Architecture (each followed by BN → ReLU):
        Conv3d(128 → 64, 3, stride=(2, 1, 1), padding=(1, 1, 1)) →
        Conv3d(64 → 64, 3, stride=(1, 1, 1), padding=(0, 1, 1)) →
        Conv3d(64 → 64, 3, stride=(2, 1, 1), padding=(1, 1, 1))

    The depth dimension is then merged into channels, e.g. for D = 10:
        (B, 128, 10, H, W) → (B, 64, 2, H, W) → (B, 128, H, W)
    """

    def __init__(self, in_channels: int = 128, channels: int = 64, depth: int = 10) -> None:
        super().__init__()

        specs = [
            (in_channels, channels, (2, 1, 1), (1, 1, 1)),
            (channels, channels, (1, 1, 1), (0, 1, 1)),
            (channels, channels, (2, 1, 1), (1, 1, 1)),
        ]

        layers = []
        out_depth = depth
        for c_in, c_out, stride, padding in specs:
            layers += [
                nn.Conv3d(c_in, c_out, kernel_size=3, stride=stride, padding=padding, bias=False),
                nn.BatchNorm3d(c_out),
                nn.ReLU(inplace=True),
            ]
            out_depth = _conv3d_out_depth(out_depth, stride[0], padding[0])

        assert out_depth > 0, f"Voxel grid depth {depth} is too small for the middle layers"

        self.layers = nn.Sequential(*layers)
        self.out_depth = out_depth
        self.out_channels = channels * out_depth

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Dense voxel features (B, C, D, H, W)

        Returns:
            BEV feature map (B, C' * D', H, W)
        """
        x = self.layers(x)
        batch_size, channels, depth, height, width = x.shape
        return x.reshape(batch_size, channels * depth, height, width)
