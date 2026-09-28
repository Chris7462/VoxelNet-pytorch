import torch.nn as nn
from torch import Tensor

from ..voxel_encoder import SVFE
from ..middle import ConvMiddleLayer
from ..backbone import RPNBackbone
from ..neck import RPNNeck
from ..head import RPNHead


class VoxelNet(nn.Module):
    """
    VoxelNet for LiDAR-based 3D object detection.

    Architecture:
        voxels (V, T, 7), num_points (V,), coords (V, 4)
            │
            ▼
        svfe ──────────────── (V, 128)                  # stacked voxel feature encoding
            │
            ▼
        scatter ───────────── (B, 128, D, H, W)         # sparse voxels → dense grid
            │
            ▼
        middle ────────────── (B, 128, H, W)            # 3D convs, depth merged into channels
            │
            ▼
        backbone ──────────── strides 2 / 4 / 8         # RPN conv blocks
            │
            ▼
        neck ──────────────── (B, 768, H/2, W/2)        # deconv + concat
            │
            ├──────────────────────────┐
            ▼                          ▼
        psm (B, A, H/2, W/2)      rm (B, 7A, H/2, W/2)

    Reference:
        "VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection"
        https://arxiv.org/abs/1711.06396
    """

    def __init__(
        self,
        grid_size: tuple[int, int, int],
        num_anchors: int = 2,
        vfe_channels: tuple[int, ...] = (32, 128),
        voxel_feature_dim: int = 128,
    ) -> None:
        """
        Args:
            grid_size: Voxel grid size (nx, ny, nz); nx and ny must be divisible by 8
            num_anchors: Anchors per feature map position
            vfe_channels: Output channels of the VFE layers
            voxel_feature_dim: Output channels of the SVFE
        """
        super().__init__()

        nx, ny, nz = grid_size
        assert nx % 8 == 0 and ny % 8 == 0, f"Grid size (nx={nx}, ny={ny}) must be divisible by 8"
        self.grid_size = (nx, ny, nz)

        self.svfe = SVFE(in_channels=7, vfe_channels=tuple(vfe_channels), out_channels=voxel_feature_dim)
        self.middle = ConvMiddleLayer(in_channels=voxel_feature_dim, depth=nz)
        self.backbone = RPNBackbone(in_channels=self.middle.out_channels)
        self.neck = RPNNeck(in_channels=self.backbone.out_channels)
        self.head = RPNHead(in_channels=self.neck.out_channels, num_anchors=num_anchors)

    def scatter(self, voxel_features: Tensor, coords: Tensor, batch_size: int) -> Tensor:
        """
        Scatter sparse voxel features into a dense grid.

        Args:
            voxel_features: (V, C)
            coords: (V, 4) as (batch, z, y, x)
            batch_size: Number of samples in the batch

        Returns:
            Dense features (B, C, D, H, W)
        """
        nx, ny, nz = self.grid_size
        channels = voxel_features.shape[1]

        dense = voxel_features.new_zeros(batch_size, nz, ny, nx, channels)
        dense[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]] = voxel_features
        return dense.permute(0, 4, 1, 2, 3).contiguous()

    def forward(
        self,
        voxels: Tensor,
        num_points: Tensor,
        coords: Tensor,
        batch_size: int,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            voxels: Point features (V, T, 7)
            num_points: Valid points per voxel (V,)
            coords: Voxel coordinates (V, 4) as (batch, z, y, x)
            batch_size: Number of samples in the batch

        Returns:
            psm: Score logits (B, A, H/2, W/2)
            rm: Regression map (B, 7A, H/2, W/2)
        """
        x = self.svfe(voxels, num_points)
        x = self.scatter(x, coords, batch_size)
        x = self.middle(x)
        x = self.backbone(x)
        x = self.neck(x)
        return self.head(x)
