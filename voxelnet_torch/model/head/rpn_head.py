import torch.nn as nn
from torch import Tensor


class RPNHead(nn.Module):
    """
    VoxelNet detection head: two 1x1 convolutions on the concatenated RPN features.

    Outputs:
        probability score map (logits): (B, A, H, W)
        regression map:                 (B, 7A, H, W), 7 values per anchor
    """

    def __init__(self, in_channels: int = 768, num_anchors: int = 2) -> None:
        super().__init__()

        self.score_head = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.reg_head = nn.Conv2d(in_channels, 7 * num_anchors, kernel_size=1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: Feature map (B, C, H, W)

        Returns:
            psm: Score logits (B, A, H, W)
            rm: Regression map (B, 7A, H, W)
        """
        return self.score_head(x), self.reg_head(x)
