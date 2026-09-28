"""
Box utilities.

Box convention (LiDAR frame): [x, y, z, h, w, l, yaw]
    x, y : BEV center
    z    : *bottom* center height (KITTI convention)
    h    : height (along z)
    w    : width  (along the box's local y axis)
    l    : length (along the box's local x axis)
    yaw  : rotation around the LiDAR z axis

NumPy functions are used by the (CPU) data pipeline, torch functions by
post-processing on the model output.
"""

import numpy as np
import torch
from torch import Tensor


def limit_period(val: np.ndarray, offset: float = 0.5, period: float = np.pi) -> np.ndarray:
    """Wrap angles into [-offset * period, (1 - offset) * period)."""
    return val - np.floor(val / period + offset) * period


def boxes_to_bev_corners(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes to their 4 bird's-eye-view corners.

    Args:
        boxes: (N, 7) boxes

    Returns:
        (N, 4, 2) corners, ordered (-l/2, +w/2), (-l/2, -w/2), (+l/2, -w/2), (+l/2, +w/2)
        before rotation.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    x, y, w, l, yaw = boxes[:, 0], boxes[:, 1], boxes[:, 4], boxes[:, 5], boxes[:, 6]

    local_x = np.stack([-l, -l, l, l], axis=1) / 2          # (N, 4)
    local_y = np.stack([w, -w, -w, w], axis=1) / 2          # (N, 4)

    cos, sin = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    corners_x = cos * local_x - sin * local_y + x[:, None]
    corners_y = sin * local_x + cos * local_y + y[:, None]

    return np.stack([corners_x, corners_y], axis=-1)


def boxes_to_corners_3d(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes to 8 corners: 4 bottom corners followed by the 4 top corners.

    Args:
        boxes: (N, 7) boxes

    Returns:
        (N, 8, 3) corners
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    bev = boxes_to_bev_corners(boxes)                                   # (N, 4, 2)
    z_bottom = np.repeat(boxes[:, None, 2], 4, axis=1)                  # (N, 4)
    z_top = z_bottom + boxes[:, None, 3]

    bottom = np.concatenate([bev, z_bottom[..., None]], axis=-1)
    top = np.concatenate([bev, z_top[..., None]], axis=-1)
    return np.concatenate([bottom, top], axis=1)


def boxes_to_standup_bev(boxes: np.ndarray) -> np.ndarray:
    """
    Axis-aligned ("standup") BEV rectangle enclosing each rotated box.

    Args:
        boxes: (N, 7) boxes

    Returns:
        (N, 4) rectangles as [x1, y1, x2, y2]
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    x, y, w, l, yaw = boxes[:, 0], boxes[:, 1], boxes[:, 4], boxes[:, 5], boxes[:, 6]
    cos, sin = np.abs(np.cos(yaw)), np.abs(np.sin(yaw))
    half_x = (l * cos + w * sin) / 2
    half_y = (l * sin + w * cos) / 2
    return np.stack([x - half_x, y - half_y, x + half_x, y + half_y], axis=1)


def iou_2d(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    IoU between two sets of axis-aligned rectangles given in metric units.

    Unlike the Faster R-CNN pixel convention, no "+1" is added to widths.

    Args:
        boxes_a: (N, 4) [x1, y1, x2, y2]
        boxes_b: (K, 4) [x1, y1, x2, y2]

    Returns:
        (N, K) IoU matrix
    """
    boxes_a = np.asarray(boxes_a, dtype=np.float64)
    boxes_b = np.asarray(boxes_b, dtype=np.float64)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]

    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def points_in_boxes(points: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Test which points lie inside which (rotated) boxes.

    Args:
        points: (P, >=3) points
        boxes: (N, 7) boxes

    Returns:
        (P, N) boolean mask
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    xyz = points[:, None, :3] - boxes[None, :, :3]                  # (P, N, 3)
    cos, sin = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])

    # Rotate points into the box's local frame
    local_x = cos * xyz[..., 0] + sin * xyz[..., 1]
    local_y = -sin * xyz[..., 0] + cos * xyz[..., 1]

    inside_x = np.abs(local_x) <= boxes[:, 5] / 2
    inside_y = np.abs(local_y) <= boxes[:, 4] / 2
    inside_z = (xyz[..., 2] >= 0) & (xyz[..., 2] <= boxes[:, 3])
    return inside_x & inside_y & inside_z


def encode_boxes(gt_boxes: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """
    Encode ground-truth boxes as regression targets w.r.t. anchors (VoxelNet paper, eq. 1).

    The yaw residual is wrapped into [-pi/2, pi/2): a BEV box is symmetric under a
    rotation of pi, so a box and its flipped version must produce the same target.

    Args:
        gt_boxes: (N, 7) matched ground-truth boxes
        anchors: (N, 7) anchors

    Returns:
        (N, 7) targets [dx, dy, dz, dh, dw, dl, dyaw]
    """
    diag = np.sqrt(anchors[:, 4] ** 2 + anchors[:, 5] ** 2)

    return np.stack([
        (gt_boxes[:, 0] - anchors[:, 0]) / diag,
        (gt_boxes[:, 1] - anchors[:, 1]) / diag,
        (gt_boxes[:, 2] - anchors[:, 2]) / anchors[:, 3],
        np.log(gt_boxes[:, 3] / anchors[:, 3]),
        np.log(gt_boxes[:, 4] / anchors[:, 4]),
        np.log(gt_boxes[:, 5] / anchors[:, 5]),
        limit_period(gt_boxes[:, 6] - anchors[:, 6]),
    ], axis=1)


def decode_boxes(deltas: Tensor, anchors: Tensor) -> Tensor:
    """
    Decode regression output back to boxes (inverse of `encode_boxes`).

    Args:
        deltas: (..., 7) regression output
        anchors: (..., 7) anchors, broadcastable to `deltas`

    Returns:
        (..., 7) boxes
    """
    diag = torch.sqrt(anchors[..., 4] ** 2 + anchors[..., 5] ** 2)

    x = deltas[..., 0] * diag + anchors[..., 0]
    y = deltas[..., 1] * diag + anchors[..., 1]
    z = deltas[..., 2] * anchors[..., 3] + anchors[..., 2]
    hwl = torch.exp(deltas[..., 3:6]) * anchors[..., 3:6]
    yaw = deltas[..., 6] + anchors[..., 6]

    return torch.cat([x[..., None], y[..., None], z[..., None], hwl, yaw[..., None]], dim=-1)


def boxes_to_standup_bev_torch(boxes: Tensor) -> Tensor:
    """Torch version of `boxes_to_standup_bev`: (N, 7) -> (N, 4) [x1, y1, x2, y2]."""
    x, y, w, l, yaw = boxes[:, 0], boxes[:, 1], boxes[:, 4], boxes[:, 5], boxes[:, 6]
    cos, sin = torch.cos(yaw).abs(), torch.sin(yaw).abs()
    half_x = (l * cos + w * sin) / 2
    half_y = (l * sin + w * cos) / 2
    return torch.stack([x - half_x, y - half_y, x + half_x, y + half_y], dim=1)
