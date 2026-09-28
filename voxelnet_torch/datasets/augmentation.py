"""
Data augmentation for point clouds and 3D boxes (VoxelNet paper, section 3.2).

All functions return new arrays and take an explicit `np.random.Generator`.
Boxes use the [x, y, z_bottom, h, w, l, yaw] convention from `utils.box_ops`.
"""

import numpy as np

from ..utils.box_ops import boxes_to_standup_bev, iou_2d, limit_period, points_in_boxes


def _rotate_xy(xy: np.ndarray, angle: float, center: np.ndarray | None = None) -> np.ndarray:
    """Rotate (N, 2) points counter-clockwise by `angle` around `center` (default: origin)."""
    if center is None:
        center = np.zeros(2)
    cos, sin = np.cos(angle), np.sin(angle)
    rot = np.array([[cos, -sin], [sin, cos]])
    return (xy - center) @ rot.T + center


def per_object_transform(
    points: np.ndarray,
    boxes: np.ndarray,
    rng: np.random.Generator,
    max_rotation: float = np.pi / 10,
    translation_std: float = 1.0,
    max_trials: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly rotate (around the box center) and translate each ground-truth box
    together with the points inside it. A transform is rejected if the moved box
    would overlap (in BEV) any other box; after `max_trials` rejections the box is
    left unchanged.

    Args:
        points: (P, 4) points
        boxes: (N, 7) boxes
        rng: Random generator
        max_rotation: Rotation is sampled from uniform(-max_rotation, max_rotation)
        translation_std: Translation is sampled from normal(0, translation_std) per axis
        max_trials: Number of attempts to find a collision-free transform

    Returns:
        Transformed (points, boxes)
    """
    points = points.copy()
    boxes = boxes.copy()

    for idx in range(len(boxes)):
        others = np.delete(boxes, idx, axis=0)
        others_standup = boxes_to_standup_bev(others)

        for _ in range(max_trials):
            d_yaw = rng.uniform(-max_rotation, max_rotation)
            d_xyz = rng.normal(0.0, translation_std, size=3)

            moved = boxes[idx].copy()
            moved[:3] += d_xyz
            moved[6] += d_yaw

            if len(others) == 0 or not np.any(iou_2d(boxes_to_standup_bev(moved[None]), others_standup) > 0):
                break
        else:
            continue

        inside = points_in_boxes(points, boxes[idx][None])[:, 0]
        center = boxes[idx, :2]
        points[inside, :2] = _rotate_xy(points[inside, :2], d_yaw, center)
        points[inside, :3] += d_xyz

        moved[6] = limit_period(moved[6], offset=0.5, period=2 * np.pi)
        boxes[idx] = moved

    return points, boxes


def global_rotation(
    points: np.ndarray,
    boxes: np.ndarray,
    rng: np.random.Generator,
    max_rotation: float = np.pi / 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the whole scene around the LiDAR z axis by uniform(-max_rotation, max_rotation)."""
    angle = rng.uniform(-max_rotation, max_rotation)

    points = points.copy()
    boxes = boxes.copy()
    points[:, :2] = _rotate_xy(points[:, :2], angle)
    boxes[:, :2] = _rotate_xy(boxes[:, :2], angle)
    boxes[:, 6] = limit_period(boxes[:, 6] + angle, offset=0.5, period=2 * np.pi)

    return points, boxes


def global_scaling(
    points: np.ndarray,
    boxes: np.ndarray,
    rng: np.random.Generator,
    scale_range: tuple[float, float] = (0.95, 1.05),
) -> tuple[np.ndarray, np.ndarray]:
    """Scale the whole scene (coordinates and box sizes) by uniform(*scale_range)."""
    factor = rng.uniform(*scale_range)

    points = points.copy()
    boxes = boxes.copy()
    points[:, :3] *= factor
    boxes[:, :6] *= factor

    return points, boxes


class Augmentor:
    """
    Apply one of the three augmentations per sample, chosen at random with the
    probabilities given in the `augmentation` section of the config.

    Args:
        config: The `augmentation` section of the config
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        probs = np.array([
            config['prob_per_object'],
            config['prob_global_rotation'],
            config['prob_global_scaling'],
        ], dtype=np.float64)
        self.probs = probs / probs.sum()

    def __call__(
        self,
        points: np.ndarray,
        boxes: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        choice = rng.choice(3, p=self.probs)

        if choice == 0:
            return per_object_transform(
                points, boxes, rng,
                max_rotation=cfg['per_object_rotation'],
                translation_std=cfg['per_object_translation_std'],
                max_trials=cfg['per_object_max_trials'],
            )
        if choice == 1:
            return global_rotation(points, boxes, rng, max_rotation=cfg['global_rotation'])
        return global_scaling(points, boxes, rng, scale_range=tuple(cfg['global_scaling']))
