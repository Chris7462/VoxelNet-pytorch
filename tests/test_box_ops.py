import numpy as np
import torch
from torchvision.ops import box_iou

from voxelnet_torch.utils import (
    boxes_to_bev_corners,
    boxes_to_corners_3d,
    boxes_to_standup_bev,
    boxes_to_standup_bev_torch,
    decode_boxes,
    encode_boxes,
    iou_2d,
    points_in_boxes,
)


def random_boxes(rng, n):
    return np.column_stack([
        rng.uniform(0, 70, n), rng.uniform(-40, 40, n), rng.uniform(-2, -1.5, n),
        rng.uniform(1.4, 1.8, n), rng.uniform(1.5, 1.9, n), rng.uniform(3.5, 4.5, n),
        rng.uniform(-np.pi, np.pi, n),
    ])


def test_iou_matches_torchvision():
    rng = np.random.default_rng(0)
    a = boxes_to_standup_bev(random_boxes(rng, 300))
    b = boxes_to_standup_bev(random_boxes(rng, 20))
    b[:5] = a[:5] + rng.normal(0, 0.3, size=(5, 4))       # guarantee some overlaps

    expected = box_iou(torch.from_numpy(a), torch.from_numpy(b)).numpy()
    np.testing.assert_allclose(iou_2d(a, b), expected, atol=1e-10)


def test_iou_has_no_plus_one():
    # Two 1.6 m x 3.9 m rectangles shifted by 2 m: IoU = 1.9 / 5.9
    a = np.array([[0.0, 0.0, 3.9, 1.6]])
    b = np.array([[2.0, 0.0, 5.9, 1.6]])
    np.testing.assert_allclose(iou_2d(a, b), [[1.9 / 5.9]])


def test_standup_closed_form_matches_corners():
    rng = np.random.default_rng(1)
    boxes = random_boxes(rng, 100)
    corners = boxes_to_bev_corners(boxes)
    expected = np.concatenate([corners.min(axis=1), corners.max(axis=1)], axis=1)

    np.testing.assert_allclose(boxes_to_standup_bev(boxes), expected, atol=1e-10)
    np.testing.assert_allclose(
        boxes_to_standup_bev_torch(torch.from_numpy(boxes)).numpy(), expected, atol=1e-10)


def test_corners_3d_heights():
    boxes = np.array([[10.0, 2.0, -1.7, 1.5, 1.6, 3.9, 0.3]])
    corners = boxes_to_corners_3d(boxes)
    np.testing.assert_allclose(corners[0, :4, 2], -1.7)
    np.testing.assert_allclose(corners[0, 4:, 2], -0.2)


def test_encode_decode_roundtrip():
    rng = np.random.default_rng(2)
    gt = random_boxes(rng, 50)
    anchors = random_boxes(rng, 50)
    anchors[:, 6] = rng.choice([0.0, np.pi / 2], size=50)

    deltas = encode_boxes(gt, anchors)
    assert (np.abs(deltas[:, 6]) <= np.pi / 2 + 1e-9).all()

    decoded = decode_boxes(torch.from_numpy(deltas), torch.from_numpy(anchors)).numpy()
    np.testing.assert_allclose(decoded[:, :6], gt[:, :6], atol=1e-9)

    # Yaw is recovered modulo pi (a BEV box is symmetric under a rotation of pi)
    yaw_diff = (decoded[:, 6] - gt[:, 6]) / np.pi
    np.testing.assert_allclose(yaw_diff, np.round(yaw_diff), atol=1e-9)


def test_points_in_boxes():
    box = np.array([[10.0, 0.0, -1.7, 1.5, 2.0, 4.0, np.pi / 2]])   # length along y after rotation
    points = np.array([
        [10.0, 1.9, -1.0],    # inside (along the rotated length)
        [11.9, 0.0, -1.0],    # outside (beyond half width 1.0)
        [10.0, 0.0, -1.8],    # below the box
        [10.0, 0.0, -0.3],    # inside, near the top
    ])
    np.testing.assert_array_equal(points_in_boxes(points, box)[:, 0], [True, False, False, True])


def test_rotated_bev_iou_and_nms():
    from voxelnet_torch.utils import rotated_bev_iou, rotated_nms, iou_2d

    box = np.array([[10.0, 0.0, -1.7, 1.5, 2.0, 4.0, 0.0]])
    rotated = box.copy()
    rotated[0, 6] = np.pi / 2
    np.testing.assert_allclose(rotated_bev_iou(box, box), [[1.0]])
    np.testing.assert_allclose(rotated_bev_iou(box, rotated), [[1 / 3]])   # 2x2 / (8 + 8 - 4)

    # Two cars parked side by side at 45 degrees: their enclosing rectangles overlap
    # (so standup NMS at 0.1 removes one), their actual footprints do not
    yaw, gap = np.pi / 4, 2.2
    c2 = np.array([20.0, 0.0]) + gap * np.array([-np.sin(yaw), np.cos(yaw)])
    boxes = np.array([[20.0, 0.0, -1.7, 1.5, 1.8, 4.2, yaw], [c2[0], c2[1], -1.7, 1.5, 1.8, 4.2, yaw]])
    assert iou_2d(boxes_to_standup_bev(boxes[:1]), boxes_to_standup_bev(boxes[1:]))[0, 0] > 0.1
    assert rotated_bev_iou(boxes[:1], boxes[1:])[0, 0] == 0.0
    assert rotated_nms(boxes, np.array([0.9, 0.8]), 0.1).tolist() == [0, 1]

    # Duplicates of the same car are suppressed, highest score kept
    dup = np.repeat(box, 3, axis=0)
    dup[:, 0] += [0.0, 0.1, 0.2]
    assert rotated_nms(dup, np.array([0.5, 0.9, 0.7]), 0.1).tolist() == [1]
    assert len(rotated_nms(np.zeros((0, 7)), np.zeros(0), 0.1)) == 0
