import numpy as np

from voxelnet_torch.datasets.augmentation import global_rotation, global_scaling, per_object_transform
from voxelnet_torch.utils import boxes_to_standup_bev, iou_2d, points_in_boxes

from conftest import make_scene as _make_scene


def make_scene(rng, num_cars):
    """Scene plus points strictly inside each box (surface points sit exactly on the boundary)."""
    points, boxes = _make_scene(rng, num_cars)
    interior = []
    for x, y, z, h, w, l, yaw in boxes:
        local = rng.uniform(-0.45, 0.45, size=(200, 3)) * [l, w, h]
        cos, sin = np.cos(yaw), np.sin(yaw)
        interior.append(np.column_stack([
            cos * local[:, 0] - sin * local[:, 1] + x,
            sin * local[:, 0] + cos * local[:, 1] + y,
            local[:, 2] + z + h / 2,
            np.zeros(200),
        ]))
    # Drop the (boundary) surface points and ground points close to a box
    tall = boxes * [1, 1, 1, 1, 1, 1, 1] + [0, 0, -1, 2, 0.5, 0.5, 0]
    near = points_in_boxes(points, tall).any(axis=1)
    return np.concatenate([points[~near]] + interior).astype(np.float32), boxes


def membership(points, boxes):
    return points_in_boxes(points, boxes)


def test_global_rotation_preserves_membership():
    rng = np.random.default_rng(0)
    points, boxes = make_scene(rng, 4)
    before = membership(points, boxes)

    new_points, new_boxes = global_rotation(points, boxes, rng, max_rotation=np.pi / 4)

    np.testing.assert_array_equal(membership(new_points, new_boxes), before)
    assert not np.allclose(new_points[:, :2], points[:, :2])
    assert (np.abs(new_boxes[:, 6]) <= np.pi).all()


def test_global_scaling_preserves_membership():
    rng = np.random.default_rng(1)
    points, boxes = make_scene(rng, 4)
    before = membership(points, boxes)

    new_points, new_boxes = global_scaling(points, boxes, rng)
    np.testing.assert_array_equal(membership(new_points, new_boxes), before)


def test_per_object_moves_points_with_box_and_avoids_collisions():
    rng = np.random.default_rng(2)
    points, boxes = make_scene(rng, 4)
    before = membership(points, boxes)

    new_points, new_boxes = per_object_transform(points, boxes, rng, max_rotation=np.pi / 10)

    # Every point that was inside a box is inside the moved box
    after = membership(new_points, new_boxes)
    assert (after[before]).all()

    # Boxes moved, but rotation is around the box center: moves stay small
    shift = np.linalg.norm(new_boxes[:, :2] - boxes[:, :2], axis=1)
    assert (shift > 0).any() and (shift < 6).all()

    # No overlaps between moved boxes
    standup = boxes_to_standup_bev(new_boxes)
    iou = iou_2d(standup, standup)
    np.fill_diagonal(iou, 0)
    assert (iou == 0).all()


def test_inputs_not_modified():
    rng = np.random.default_rng(3)
    points, boxes = make_scene(rng, 3)
    points_copy, boxes_copy = points.copy(), boxes.copy()
    for fn in (per_object_transform, global_rotation, global_scaling):
        fn(points, boxes, rng)
    np.testing.assert_array_equal(points, points_copy)
    np.testing.assert_array_equal(boxes, boxes_copy)
