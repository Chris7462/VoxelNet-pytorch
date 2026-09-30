"""GT database creation and GT sampling augmentation."""

import os
import pickle
import sys

import numpy as np
import pytest

from voxelnet_torch.datasets import GTSampler, KITTI
from voxelnet_torch.datasets.kitti_io import read_calib, read_label, read_lidar
from voxelnet_torch.utils import points_in_boxes, rotated_bev_iou

from conftest import REPO_ROOT, small_config

sys.path.insert(0, REPO_ROOT)
from tools.create_gt_database import create_gt_database  # noqa: E402

# The synthetic labels have empty 2D boxes (difficulty -1), so do not filter on difficulty
def grown(boxes, margin=0.01):
    """Boxes enlarged by a small margin: synthetic points lie exactly on the box surfaces."""
    boxes = np.array(boxes, dtype=np.float64).reshape(-1, 7)
    boxes[:, 2] -= margin
    boxes[:, 3:6] += 2 * margin
    return boxes


SAMPLING = {'enabled': True, 'database_dir': 'gt_database', 'sample_groups': {'Car': 12},
            'min_points': {'Car': 5}, 'filter_difficulty': []}


@pytest.fixture
def database(kitti_root):
    root, gt = kitti_root
    counts = create_gt_database(root, 'crop', 'train', 'gt_database', ['Car', 'Van'])
    return root, gt, counts


def test_create_gt_database(database):
    root, gt, counts = database
    # Train split: frames 000000-000003, 3 cars each except the empty frame 000003
    assert counts == {'Car': 9}

    with open(os.path.join(root, 'gt_database', 'infos.pkl'), 'rb') as f:
        infos = pickle.load(f)['infos']
    points = np.load(os.path.join(root, 'gt_database', 'points.npy'))
    assert {info['frame_id'] for info in infos} == {'000000', '000001', '000002'}

    for info in infos:
        obj = points[info['offset']:info['offset'] + info['num_points']]
        assert info['num_points'] > 50                     # ~300 surface points per synthetic car
        assert points_in_boxes(obj, grown(info['box'])).all()
        matches = np.abs(gt[info['frame_id']] - info['box']).max(axis=1)
        assert matches.min() < 1e-3


def test_sampler_adds_non_overlapping_objects(database):
    root, _, _ = database
    sampler = GTSampler(root, SAMPLING)
    assert sampler.pool_size('Car') == 9

    # Scene: frame 000004 (val split, 3 cars)
    calib = read_calib(os.path.join(root, 'training', 'calib', '000004.txt'))
    label = os.path.join(root, 'training', 'label_2', '000004.txt')
    points = read_lidar(os.path.join(root, 'training', 'crop', '000004.bin'))
    boxes = read_label(label, calib, ['Car', 'Van'])

    rng = np.random.default_rng(0)
    new_points, new_boxes = sampler(points, boxes, np.zeros((0, 7)), rng)

    assert len(boxes) < len(new_boxes) <= 12
    np.testing.assert_array_equal(new_boxes[:len(boxes)], boxes)     # original boxes first, unchanged

    # No two boxes overlap in bird's-eye view
    iou = rotated_bev_iou(new_boxes, new_boxes)
    np.fill_diagonal(iou, 0)
    assert (iou < 1e-6).all()

    # New cloud = scene points outside the sampled boxes, followed by the pasted object points
    sampled = new_boxes[len(boxes):]
    with open(os.path.join(root, 'gt_database', 'infos.pkl'), 'rb') as f:
        infos = pickle.load(f)['infos']
    expected = sum(info['num_points'] for info in infos
                   if np.abs(sampled - info['box']).max(axis=1).min() < 1e-5)
    outside = ~points_in_boxes(points, sampled).any(axis=1)
    assert len(new_points) == outside.sum() + expected
    np.testing.assert_array_equal(new_points[:outside.sum()], points[outside])
    assert points_in_boxes(new_points[outside.sum():], grown(sampled)).any(axis=1).all()


def test_sampler_respects_other_objects_and_target(database):
    root, _, _ = database
    sampler = GTSampler(root, SAMPLING)
    points = np.zeros((0, 4), dtype=np.float32)
    rng = np.random.default_rng(1)

    # Scene already has enough objects: nothing is added
    many = np.tile(np.array([[5.0, 0, -1.7, 1.5, 1.6, 3.9, 0]]), (12, 1))
    _, out = sampler(points, many, np.zeros((0, 7)), rng)
    assert len(out) == 12

    # A large "other" object covering the whole range blocks every sample
    blocker = np.array([[12.8, 0.0, -3.0, 4.0, 60.0, 60.0, 0.0]])
    _, out = sampler(points, np.zeros((0, 7), dtype=np.float32), blocker, rng)
    assert len(out) == 0


def test_training_samples_with_gt_sampling(database):
    root, _, _ = database
    config = small_config(root)
    config['augmentation']['gt_sampling'] = SAMPLING

    plain = KITTI(small_config(root), 'train', training=True)
    sampled = KITTI(config, 'train', training=True)
    assert plain.gt_sampler is None and sampled.gt_sampler is not None
    assert KITTI(config, 'val', training=False).gt_sampler is None          # never at validation

    counts = [len(sampled[i]['gt_boxes']) for i in range(len(sampled))]
    base = [len(plain[i]['gt_boxes']) for i in range(len(plain))]
    assert sum(counts) > sum(base)
    sample = sampled[3]                                                    # empty frame gets objects
    assert len(sample['gt_boxes']) > 0 and sample['pos_equal_one'].sum() > 0
    assert np.isfinite(sample['targets']).all()


def test_shipped_gt_sampling_config():
    from voxelnet_torch.utils import load_config
    config = load_config(os.path.join(REPO_ROOT, 'configs', 'voxelnet_kitti_car_2gpu_bs16_cosine_gtsample.yaml'))
    gt_cfg = config['augmentation']['gt_sampling']
    assert gt_cfg['enabled'] and gt_cfg['sample_groups'] == {'Car': 15}
