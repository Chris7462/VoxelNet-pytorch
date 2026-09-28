import numpy as np

from .config import compute_grid_size


def generate_anchors(
    point_cloud_range: list[float],
    voxel_size: list[float],
    size: list[float],
    z_center: float,
    rotations: list[float],
    stride: int = 2,
) -> np.ndarray:
    """
    Generate anchors centered on each cell of the RPN output feature map.

    The RPN output has stride 2 w.r.t. the voxel grid, so there is one
    anchor set per (ny / 2, nx / 2) cell.

    Args:
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        voxel_size: [vx, vy, vz]
        size: anchor [h, w, l]
        z_center: height of the anchor center (converted to bottom-center internally)
        rotations: anchor yaw angles, one anchor per rotation
        stride: feature map stride w.r.t. the voxel grid

    Returns:
        (ny / stride, nx / stride, num_rotations, 7) anchors as [x, y, z_bottom, h, w, l, yaw]
    """
    nx, ny, _ = compute_grid_size(point_cloud_range, voxel_size)
    x_min, y_min = point_cloud_range[0], point_cloud_range[1]
    vx, vy = voxel_size[0], voxel_size[1]

    fx, fy = nx // stride, ny // stride
    xs = x_min + (np.arange(fx) + 0.5) * vx * stride
    ys = y_min + (np.arange(fy) + 0.5) * vy * stride
    cy, cx = np.meshgrid(ys, xs, indexing='ij')                     # (fy, fx)

    h, w, l = size
    num_rot = len(rotations)

    anchors = np.zeros((fy, fx, num_rot, 7), dtype=np.float32)
    anchors[..., 0] = cx[..., None]
    anchors[..., 1] = cy[..., None]
    anchors[..., 2] = z_center - h / 2
    anchors[..., 3] = h
    anchors[..., 4] = w
    anchors[..., 5] = l
    anchors[..., 6] = np.asarray(rotations, dtype=np.float32)

    return anchors


def build_anchors(config: dict) -> np.ndarray:
    """Build anchors from the config dictionary. See `generate_anchors`."""
    return generate_anchors(
        point_cloud_range=config['dataset']['point_cloud_range'],
        voxel_size=config['dataset']['voxel_size'],
        size=config['anchor']['size'],
        z_center=config['anchor']['z_center'],
        rotations=config['anchor']['rotations'],
    )
