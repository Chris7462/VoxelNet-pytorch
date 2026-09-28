"""Readers for the KITTI object detection file formats."""

import numpy as np

from ..utils.box_ops import limit_period


def read_lidar(lidar_file: str) -> np.ndarray:
    """
    Read a KITTI velodyne .bin file.

    Returns:
        (P, 4) float32 array of [x, y, z, reflectance]
    """
    return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)


def read_calib(calib_file: str) -> dict:
    """
    Read a KITTI calibration file.

    Returns:
        Dict with
            'P2': (3, 4) left color camera projection matrix
            'R0_rect': (4, 4) rectifying rotation (homogeneous)
            'Tr_velo_to_cam': (4, 4) LiDAR to reference camera transform (homogeneous)
    """
    values = {}
    with open(calib_file) as f:
        for line in f:
            if ':' not in line:
                continue
            key, data = line.split(':', 1)
            values[key.strip()] = np.array([float(v) for v in data.split()], dtype=np.float64)

    R0_rect = np.eye(4)
    R0_rect[:3, :3] = values['R0_rect'].reshape(3, 3)

    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[:3, :4] = values['Tr_velo_to_cam'].reshape(3, 4)

    return {
        'P2': values['P2'].reshape(3, 4),
        'R0_rect': R0_rect,
        'Tr_velo_to_cam': Tr_velo_to_cam,
    }


def rect_to_velo(points_rect: np.ndarray, calib: dict) -> np.ndarray:
    """
    Transform points from the rectified camera frame to the LiDAR frame.

    Args:
        points_rect: (N, 3) points in the rectified camera frame
        calib: Calibration dict from `read_calib`

    Returns:
        (N, 3) points in the LiDAR frame
    """
    velo_to_rect = calib['R0_rect'] @ calib['Tr_velo_to_cam']
    rect_to_velo_mat = np.linalg.inv(velo_to_rect)

    points = np.hstack([points_rect, np.ones((len(points_rect), 1))])
    return (points @ rect_to_velo_mat.T)[:, :3]


def velo_to_image(points_velo: np.ndarray, calib: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Project LiDAR points into the left color image.

    Args:
        points_velo: (N, 3) points in the LiDAR frame
        calib: Calibration dict from `read_calib`

    Returns:
        uv: (N, 2) pixel coordinates (u = column, v = row)
        depth: (N,) depth in the rectified camera frame (positive in front of the camera)
    """
    points = np.hstack([points_velo, np.ones((len(points_velo), 1))])
    points_rect = points @ (calib['R0_rect'] @ calib['Tr_velo_to_cam']).T     # (N, 4)
    projected = points_rect @ calib['P2'].T                                    # (N, 3)

    depth = projected[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = projected[:, :2] / depth[:, None]
    return uv, points_rect[:, 2]


def read_label(label_file: str, calib: dict, classes: list[str]) -> np.ndarray:
    """
    Read a KITTI label file and convert the boxes of the requested classes to the LiDAR frame.

    KITTI stores [h, w, l, x, y, z, ry] with (x, y, z) the bottom center in the
    rectified camera frame and ry the rotation around the camera y axis.

    Args:
        label_file: Path to the label file
        calib: Calibration dict from `read_calib`
        classes: Object classes to keep (e.g. ['Car', 'Van'])

    Returns:
        (N, 7) float32 boxes [x, y, z_bottom, h, w, l, yaw] in the LiDAR frame
    """
    rows = []
    with open(label_file) as f:
        for line in f:
            fields = line.split()
            if not fields or fields[0] not in classes:
                continue
            rows.append([float(v) for v in fields[8:15]])

    if not rows:
        return np.zeros((0, 7), dtype=np.float32)

    rows = np.array(rows, dtype=np.float64)
    h, w, l = rows[:, 0], rows[:, 1], rows[:, 2]
    location_velo = rect_to_velo(rows[:, 3:6], calib)
    yaw = limit_period(-rows[:, 6] - np.pi / 2, offset=0.5, period=2 * np.pi)

    boxes = np.column_stack([location_velo, h, w, l, yaw])
    return boxes.astype(np.float32)
