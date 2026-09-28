"""Shared fixtures: a small config and a synthetic KITTI dataset on disk."""

import copy
import os

import numpy as np
import pytest

from voxelnet_torch.datasets.kitti_io import read_calib
from voxelnet_torch.utils import load_config
from voxelnet_torch.utils.box_ops import limit_period

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO_ROOT, 'configs', 'voxelnet_kitti_car.yaml')

# Calibration of KITTI training frame 000000
CALIB_TEXT = """P0: 7.070493e+02 0.000000e+00 6.040814e+02 0.000000e+00 0.000000e+00 7.070493e+02 1.805066e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P1: 7.070493e+02 0.000000e+00 6.040814e+02 -3.797842e+02 0.000000e+00 7.070493e+02 1.805066e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P2: 7.070493e+02 0.000000e+00 6.040814e+02 4.575831e+01 0.000000e+00 7.070493e+02 1.805066e+02 -3.454157e-01 0.000000e+00 0.000000e+00 1.000000e+00 4.981016e-03
P3: 7.070493e+02 0.000000e+00 6.040814e+02 -3.341081e+02 0.000000e+00 7.070493e+02 1.805066e+02 2.330660e+00 0.000000e+00 0.000000e+00 1.000000e+00 3.201153e-03
R0_rect: 9.999128e-01 1.009263e-02 -8.511932e-03 -1.012729e-02 9.999406e-01 -4.037671e-03 8.470675e-03 4.123522e-03 9.999556e-01
Tr_velo_to_cam: 6.927964e-03 -9.999722e-01 -2.757829e-03 -2.457729e-02 -1.162982e-03 2.749836e-03 -9.999955e-01 -6.127237e-02 9.999753e-01 6.931141e-03 -1.143899e-03 -3.321029e-01
Tr_imu_to_velo: 9.999976e-01 7.553071e-04 -2.035826e-03 -8.086759e-01 -7.854027e-04 9.998898e-01 -1.482298e-02 3.195559e-01 2.024406e-03 1.482454e-02 9.998881e-01 -7.997231e-01
"""


def small_config(tmp_root: str | None = None) -> dict:
    """The shipped config shrunk to a 25.6 m x 25.6 m range so tests run fast on CPU."""
    config = copy.deepcopy(load_config(CONFIG_PATH))
    config['dataset']['point_cloud_range'] = [0.0, -12.8, -3.0, 25.6, 12.8, 1.0]
    config['dataloader']['num_workers'] = 0
    config['train']['amp'] = False
    if tmp_root is not None:
        config['dataset']['root'] = tmp_root
    return config


def box_to_label_line(box: np.ndarray, calib: dict, cls: str = 'Car') -> str:
    """Write a LiDAR-frame box [x, y, z_bottom, h, w, l, yaw] as a KITTI label line."""
    x, y, z, h, w, l, yaw = box
    velo_to_rect = calib['R0_rect'] @ calib['Tr_velo_to_cam']
    loc = velo_to_rect @ np.array([x, y, z, 1.0])
    ry = limit_period(-yaw - np.pi / 2, offset=0.5, period=2 * np.pi)
    return (f"{cls} 0.00 0 0.00 0.00 0.00 0.00 0.00 "
            f"{h:.6f} {w:.6f} {l:.6f} {loc[0]:.6f} {loc[1]:.6f} {loc[2]:.6f} {ry:.6f}")


def sample_box_surface(box: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample points on the sides and top of a box (what a LiDAR would see)."""
    x, y, z, h, w, l, yaw = box
    local = rng.uniform(-0.5, 0.5, size=(n, 3)) * [l, w, h]
    face = rng.integers(0, 3, size=n)
    local[face == 0, 0] = np.sign(local[face == 0, 0]) * l / 2
    local[face == 1, 1] = np.sign(local[face == 1, 1]) * w / 2
    local[face == 2, 2] = h / 2
    cos, sin = np.cos(yaw), np.sin(yaw)
    pts = np.empty((n, 4), dtype=np.float32)
    pts[:, 0] = cos * local[:, 0] - sin * local[:, 1] + x
    pts[:, 1] = sin * local[:, 0] + cos * local[:, 1] + y
    pts[:, 2] = local[:, 2] + z + h / 2
    pts[:, 3] = rng.uniform(0, 1, size=n)
    return pts


def make_scene(rng: np.random.Generator, num_cars: int, x_max: float = 24.0, y_abs: float = 11.0):
    """Random non-overlapping cars on a ground plane at z = -1.73."""
    boxes = []
    while len(boxes) < num_cars:
        box = np.array([rng.uniform(4, x_max), rng.uniform(-y_abs, y_abs), -1.73,
                        rng.uniform(1.4, 1.7), rng.uniform(1.5, 1.8), rng.uniform(3.5, 4.5),
                        rng.uniform(-np.pi, np.pi)])
        if all(np.hypot(*(box[:2] - b[:2])) > 6.0 for b in boxes):
            boxes.append(box)
    boxes = np.array(boxes).reshape(-1, 7)

    ground = np.column_stack([
        rng.uniform(0, x_max + 2, 4000), rng.uniform(-y_abs - 1, y_abs + 1, 4000),
        rng.normal(-1.73, 0.02, 4000), rng.uniform(0, 1, 4000),
    ]).astype(np.float32)
    cars = [sample_box_surface(b, 300, rng) for b in boxes]
    points = np.concatenate([ground] + cars).astype(np.float32)
    return points, boxes


@pytest.fixture
def kitti_root(tmp_path):
    """
    A synthetic KITTI root with 6 frames (4 train, 2 val).
    Frame 000003 has no cars, to exercise the empty-GT path.
    """
    rng = np.random.default_rng(0)
    root = tmp_path / 'KITTI'
    for sub in ('calib', 'label_2', 'crop'):
        (root / 'training' / sub).mkdir(parents=True)
    (root / 'ImageSets').mkdir()

    frame_ids = [f'{i:06d}' for i in range(6)]
    gt = {}
    for i, frame_id in enumerate(frame_ids):
        calib_file = root / 'training' / 'calib' / f'{frame_id}.txt'
        calib_file.write_text(CALIB_TEXT)
        calib = read_calib(str(calib_file))

        points, boxes = make_scene(rng, num_cars=0 if i == 3 else 3)
        points.tofile(root / 'training' / 'crop' / f'{frame_id}.bin')
        lines = [box_to_label_line(b, calib) for b in boxes]
        lines.append("DontCare -1 -1 -10 0 0 10 10 -1 -1 -1 -1000 -1000 -1000 -10")
        (root / 'training' / 'label_2' / f'{frame_id}.txt').write_text('\n'.join(lines) + '\n')
        gt[frame_id] = boxes

    (root / 'ImageSets' / 'train.txt').write_text('\n'.join(frame_ids[:4]) + '\n')
    (root / 'ImageSets' / 'val.txt').write_text('\n'.join(frame_ids[4:]) + '\n')
    return str(root), gt
