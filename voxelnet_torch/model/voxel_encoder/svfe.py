import torch
import torch.nn as nn
from torch import Tensor


class PointwiseFCN(nn.Module):
    """
    Point-wise fully connected layer: Linear → BN → ReLU.

    Operates on the valid (non-padded) points only, given as a flat (M, C) tensor,
    so padded points neither leak into BatchNorm statistics nor into the max-pooling.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x: Point features (V, T, C_in)
            mask: Valid point mask (V, T)

        Returns:
            Point features (V, T, C_out), zero at padded points
        """
        valid = self.layers(x[mask])
        out = valid.new_zeros(*mask.shape, valid.shape[-1])
        out[mask] = valid
        return out


class VFE(nn.Module):
    """
    Voxel Feature Encoding layer.

    Architecture:
        point-wise FCN (C_in → C_out/2) →
        element-wise max over the voxel's points (locally aggregated feature) →
        concat [point-wise, aggregated] → (C_out)
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        assert out_channels % 2 == 0, "VFE output channels must be even"

        self.fcn = PointwiseFCN(in_channels, out_channels // 2)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            x: Point features (V, T, C_in)
            mask: Valid point mask (V, T)

        Returns:
            Point features (V, T, C_out), zero at padded points
        """
        pointwise = self.fcn(x, mask)                           # (V, T, C/2), >= 0
        aggregated = pointwise.max(dim=1, keepdim=True).values  # padded points are 0, safe after ReLU
        aggregated = aggregated.expand_as(pointwise)

        out = torch.cat([pointwise, aggregated], dim=2)
        return out * mask.unsqueeze(-1).to(out.dtype)


class SVFE(nn.Module):
    """
    Stacked Voxel Feature Encoding (feature learning network).

    Architecture:
        VFE(7 → 32) → VFE(32 → 128) → FCN(128 → 128) → max over points

    Input:
        voxels (V, T, 7), num_points (V,)
    Output:
        voxel features (V, 128)
    """

    def __init__(
        self,
        in_channels: int = 7,
        vfe_channels: tuple[int, ...] = (32, 128),
        out_channels: int = 128,
    ) -> None:
        super().__init__()

        layers = []
        c_in = in_channels
        for c_out in vfe_channels:
            layers.append(VFE(c_in, c_out))
            c_in = c_out
        self.vfe_layers = nn.ModuleList(layers)

        self.fcn = PointwiseFCN(c_in, out_channels)
        self.out_channels = out_channels

    def forward(self, voxels: Tensor, num_points: Tensor) -> Tensor:
        """
        Args:
            voxels: Point features (V, T, 7)
            num_points: Number of valid points per voxel (V,)

        Returns:
            Voxel features (V, C_out)
        """
        max_points = voxels.shape[1]
        mask = torch.arange(max_points, device=voxels.device)[None, :] < num_points[:, None]

        x = voxels
        for vfe in self.vfe_layers:
            x = vfe(x, mask)

        x = self.fcn(x, mask)
        return x.max(dim=1).values
