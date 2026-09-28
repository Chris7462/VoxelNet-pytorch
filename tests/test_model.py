import numpy as np
import torch

from voxelnet_torch.datasets import Voxelizer
from voxelnet_torch.model import VoxelNet
from voxelnet_torch.model.loss import VoxelNetLoss
from voxelnet_torch.model.voxel_encoder import SVFE
from voxelnet_torch.utils import compute_grid_size

from conftest import make_scene, small_config


def voxelize_batch(config, batch_size, seed=0):
    rng = np.random.default_rng(seed)
    data_cfg = config['dataset']
    voxelizer = Voxelizer(data_cfg['point_cloud_range'], data_cfg['voxel_size'], data_cfg['max_points_per_voxel'])
    voxels, nums, coords = [], [], []
    for i in range(batch_size):
        points, _ = make_scene(rng, 3)
        low, high = np.array(data_cfg['point_cloud_range'][:3]), np.array(data_cfg['point_cloud_range'][3:])
        points = points[np.all((points[:, :3] >= low) & (points[:, :3] < high), axis=1)]
        v, n, c = voxelizer(points)
        voxels.append(v)
        nums.append(n)
        coords.append(np.pad(c, ((0, 0), (1, 0)), constant_values=i))
    return (torch.from_numpy(np.concatenate(voxels)), torch.from_numpy(np.concatenate(nums)).long(),
            torch.from_numpy(np.concatenate(coords)).long())


def build(config):
    grid = compute_grid_size(config['dataset']['point_cloud_range'], config['dataset']['voxel_size'])
    return VoxelNet(grid, num_anchors=2), grid


def test_output_shapes_and_backward():
    torch.manual_seed(0)
    config = small_config()
    model, (nx, ny, nz) = build(config)
    voxels, num_points, coords = voxelize_batch(config, batch_size=2)

    psm, rm = model(voxels, num_points, coords, batch_size=2)
    assert psm.shape == (2, 2, ny // 2, nx // 2)
    assert rm.shape == (2, 14, ny // 2, nx // 2)

    B, A, H, W = psm.shape
    pos = torch.zeros(B, H, W, A)
    pos[0, 10, 10, 0] = 1
    neg = 1 - pos
    targets = torch.randn(B, H, W, 7 * A) * 0.1
    loss, loss_cls, loss_reg = VoxelNetLoss()(psm, rm, pos, neg, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_full_kitti_grid_arithmetic():
    """The shipped config gives a 128-channel BEV map and a (200, 176) output grid."""
    from voxelnet_torch.utils import load_config
    from conftest import CONFIG_PATH
    model, grid = build(load_config(CONFIG_PATH))
    assert grid == (352, 400, 10)
    assert model.middle.out_depth == 2 and model.middle.out_channels == 128
    assert model.neck.out_channels == 768


def test_scatter_places_features():
    config = small_config()
    model, (nx, ny, nz) = build(config)
    feats = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3)
    coords = torch.tensor([[0, 1, 2, 3], [1, 4, 5, 6]])
    dense = model.scatter(feats, coords, batch_size=2)
    assert dense.shape == (2, 3, nz, ny, nx)
    assert torch.equal(dense[0, :, 1, 2, 3], feats[0]) and torch.equal(dense[1, :, 4, 5, 6], feats[1])
    assert dense.sum() == feats.sum()


def test_svfe_ignores_padding():
    """Garbage in padded slots must not change the voxel features (and BN stats)."""
    torch.manual_seed(0)
    svfe = SVFE().train()
    voxels = torch.randn(50, 35, 7)
    num_points = torch.randint(1, 36, (50,))
    mask = torch.arange(35)[None] < num_points[:, None]
    clean = voxels * mask[..., None]

    out_clean = svfe(clean, num_points)
    out_dirty = svfe(clean + torch.randn_like(clean) * 5 * (~mask[..., None]), num_points)
    torch.testing.assert_close(out_clean, out_dirty)


def test_loss_matches_original_formula():
    """Same values as the original sigmoid + log implementation (up to its 1e-6 epsilon)."""
    torch.manual_seed(0)
    B, A, H, W = 2, 2, 8, 8
    psm = torch.randn(B, A, H, W)
    rm = torch.randn(B, 7 * A, H, W)
    pos = (torch.rand(B, H, W, A) > 0.9).float()
    neg = (1 - pos) * (torch.rand(B, H, W, A) > 0.3).float()
    targets = torch.randn(B, H, W, 7 * A)

    loss, loss_cls, loss_reg = VoxelNetLoss(alpha=1.5, beta=1.0)(psm, rm, pos, neg, targets)

    # Original loss.py, rewritten with non-deprecated calls
    p_pos = torch.sigmoid(psm.permute(0, 2, 3, 1))
    rm_ = rm.permute(0, 2, 3, 1).contiguous().view(B, H, W, -1, 7)
    t_ = targets.view(B, H, W, -1, 7)
    pos_reg = pos.unsqueeze(-1).expand(-1, -1, -1, -1, 7)
    cls_pos = (-pos * torch.log(p_pos + 1e-6)).sum() / (pos.sum() + 1e-6)
    cls_neg = (-neg * torch.log(1 - p_pos + 1e-6)).sum() / (neg.sum() + 1e-6)
    reg = torch.nn.functional.smooth_l1_loss(rm_ * pos_reg, t_ * pos_reg, reduction='sum') / (pos.sum() + 1e-6)

    torch.testing.assert_close(loss_cls, 1.5 * cls_pos + cls_neg, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(loss_reg, reg, rtol=1e-4, atol=1e-4)
