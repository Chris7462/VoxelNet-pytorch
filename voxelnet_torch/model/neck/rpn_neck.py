import torch
import torch.nn as nn
from torch import Tensor


def deconv_bn_relu(in_channels: int, out_channels: int, scale: int) -> nn.Sequential:
    """ConvTranspose2d(kernel=scale, stride=scale) → BN → ReLU."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale, stride=scale, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class RPNNeck(nn.Module):
    """
    Upsample the three RPN backbone outputs to stride 2 and concatenate them.

    Architecture:
        Block 1 (128, H/2) → Deconv(128 → 256, ×1) ─┐
        Block 2 (128, H/4) → Deconv(128 → 256, ×2) ─┼→ concat → (B, 768, H/2, W/2)
        Block 3 (256, H/8) → Deconv(256 → 256, ×4) ─┘
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int] = (128, 128, 256),
        out_channels: int = 256,
    ) -> None:
        super().__init__()

        self.deconv_1 = deconv_bn_relu(in_channels[0], out_channels, scale=1)
        self.deconv_2 = deconv_bn_relu(in_channels[1], out_channels, scale=2)
        self.deconv_3 = deconv_bn_relu(in_channels[2], out_channels, scale=4)

        self.out_channels = 3 * out_channels

    def forward(self, features: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """
        Args:
            features: Backbone outputs at strides 2, 4 and 8

        Returns:
            Concatenated feature map (B, 768, H/2, W/2)
        """
        x1, x2, x3 = features
        return torch.cat([self.deconv_3(x3), self.deconv_2(x2), self.deconv_1(x1)], dim=1)
