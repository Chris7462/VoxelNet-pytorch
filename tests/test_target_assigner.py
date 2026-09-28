import numpy as np
import torch

from voxelnet_torch.datasets import TargetAssigner
from voxelnet_torch.utils import build_anchors, decode_boxes, load_config

from conftest import CONFIG_PATH


def make_assigner():
    config = load_config(CONFIG_PATH)
    anchors = build_anchors(config)
    return TargetAssigner(anchors, config['anchor']['pos_iou'], config['anchor']['neg_iou']), anchors


def test_empty_gt_all_negative():
    assigner, anchors = make_assigner()
    pos, neg, targets = assigner(np.zeros((0, 7)))
    assert pos.shape == anchors.shape[:3] and targets.shape == anchors.shape[:2] + (14,)
    assert pos.sum() == 0 and neg.all() and not targets.any()


def test_gt_on_anchor():
    assigner, anchors = make_assigner()
    gt = anchors[100, 50, 1].astype(np.float64)[None].copy()      # exactly an anchor (yaw = pi/2)

    pos, neg, targets = assigner(gt)

    assert pos[100, 50, 1] == 1 and neg[100, 50, 1] == 0
    np.testing.assert_allclose(targets[100, 50, 7:14], 0, atol=1e-6)
    assert not np.any((pos == 1) & (neg == 1))


def test_targets_decode_to_gt():
    assigner, anchors = make_assigner()
    gt = np.array([
        [20.3, 5.1, -1.73, 1.52, 1.65, 4.1, 0.2],
        [45.7, -12.4, -1.68, 1.48, 1.7, 3.8, -1.4],
        [8.1, 20.0, -1.75, 1.6, 1.62, 4.3, 2.9],
    ])
    pos, neg, targets = assigner(gt)
    assert pos.sum() >= len(gt)

    mask = pos.reshape(-1).astype(bool)
    decoded = decode_boxes(
        torch.from_numpy(targets.reshape(-1, 7)[mask]).double(),
        torch.from_numpy(anchors.reshape(-1, 7)[mask]).double(),
    ).numpy()

    # Every positive anchor decodes to one of the GT boxes (yaw modulo pi)
    for box in decoded:
        dist = np.abs(gt[:, :6] - box[:6]).max(axis=1)
        match = dist.argmin()
        assert dist[match] < 1e-4
        yaw_diff = (box[6] - gt[match, 6]) / np.pi
        assert abs(yaw_diff - round(yaw_diff)) < 1e-4


def test_every_gt_gets_an_anchor():
    """A small or oddly placed GT still gets its best anchor as positive."""
    assigner, _ = make_assigner()
    gt = np.array([[30.05, 0.05, -1.73, 1.5, 0.8, 1.0, 0.785]])      # tiny box, low IoU everywhere
    pos, _, _ = assigner(gt)
    assert pos.sum() == 1
