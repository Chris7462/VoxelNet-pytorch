"""End-to-end evaluation: dataset → (perfect) model output → decode + NMS → KITTI files → AP."""

import os
import subprocess
import sys

import cv2
import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from voxelnet_torch.datasets import KITTI
from voxelnet_torch.datasets.kitti_io import (
    lidar_boxes_to_annotations, read_calib, read_label_annotations, write_annotations,
)
from voxelnet_torch.engine import Evaluator

from conftest import CALIB_TEXT, REPO_ROOT, sample_box_surface, small_config

IMAGE_SHAPE = (375, 1242)


@pytest.fixture
def eval_root(tmp_path):
    """12 val frames with 4 cars each, all inside the camera view and the (small) range."""
    rng = np.random.default_rng(0)
    root = tmp_path / 'KITTI'
    for sub in ('calib', 'label_2', 'crop', 'image_2'):
        (root / 'training' / sub).mkdir(parents=True)
    (root / 'ImageSets').mkdir()

    frame_ids = [f'{i:06d}' for i in range(12)]
    image = np.zeros((*IMAGE_SHAPE, 3), dtype=np.uint8)
    for frame_id in frame_ids:
        calib_file = root / 'training' / 'calib' / f'{frame_id}.txt'
        calib_file.write_text(CALIB_TEXT)
        calib = read_calib(str(calib_file))
        cv2.imwrite(str(root / 'training' / 'image_2' / f'{frame_id}.png'), image)

        boxes = []
        while len(boxes) < 4:
            x = rng.uniform(8, 23)
            box = np.array([x, rng.uniform(-0.35, 0.35) * x, -1.73, 1.5, 1.6, 3.9, rng.uniform(-np.pi, np.pi)])
            if all(np.hypot(*(box[:2] - b[:2])) > 6 for b in boxes):
                boxes.append(box)
        boxes = np.array(boxes)

        points = np.concatenate([sample_box_surface(b, 300, rng) for b in boxes]).astype(np.float32)
        points.tofile(root / 'training' / 'crop' / f'{frame_id}.bin')

        # Labels with the projected 2D boxes (score column dropped)
        annos = lidar_boxes_to_annotations(boxes, np.ones(len(boxes)), calib, IMAGE_SHAPE)
        assert len(annos['name']) == 4
        label_file = root / 'training' / 'label_2' / f'{frame_id}.txt'
        write_annotations(annos, str(label_file))
        lines = [' '.join(line.split()[:15]).replace('-1 -1', '0.00 0', 1) for line in label_file.read_text().splitlines()]
        label_file.write_text('\n'.join(lines) + '\n')

    (root / 'ImageSets' / 'val.txt').write_text('\n'.join(frame_ids) + '\n')
    return str(root)


class PerfectModel(torch.nn.Module):
    """Returns, batch by batch, logits and regression maps built from the training targets."""

    def __init__(self, outputs):
        super().__init__()
        self.outputs = list(outputs)

    def forward(self, voxels, num_points, coords, batch_size):
        return self.outputs.pop(0)


def perfect_outputs(config, batch_size):
    dataset = KITTI(config, 'val', training=False, assign_targets=True)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=KITTI.collate)
    outputs = []
    for batch in loader:
        pos = batch['pos_equal_one']
        psm = (pos * 20 - 10).permute(0, 3, 1, 2)
        rm = batch['targets'].permute(0, 3, 1, 2)
        outputs.append((psm, rm))
    return outputs


def test_evaluator_perfect_predictions(eval_root, tmp_path):
    config = small_config(eval_root)
    config['dataloader']['batch_size'] = 2

    dataset = KITTI(config, 'val', training=False, assign_targets=False)
    assert 'targets' not in dataset[0]
    loader = DataLoader(dataset, batch_size=2, collate_fn=KITTI.collate)

    model = PerfectModel(perfect_outputs(config, 2))
    evaluator = Evaluator(model, loader, config, torch.device('cpu'), tmp_path / 'eval')
    frame_ids = evaluator.predict()
    assert len(frame_ids) == 12 and len(os.listdir(tmp_path / 'eval' / 'predictions')) == 12

    # Written predictions match the labels
    pred = read_label_annotations(str(tmp_path / 'eval' / 'predictions' / '000000.txt'))
    gt = read_label_annotations(os.path.join(eval_root, 'training', 'label_2', '000000.txt'))
    assert 'score' in pred and len(pred['name']) == 4
    order = [np.argmin(np.abs(gt['location'] - loc).sum(axis=1)) for loc in pred['location']]
    np.testing.assert_allclose(pred['location'], gt['location'][order], atol=1e-3)
    np.testing.assert_allclose(pred['bbox'], gt['bbox'][order], atol=0.02)

    results = evaluator.evaluate(frame_ids)
    for key in results['Car']:
        for metric in ('bbox', 'bev', '3d'):
            assert min(results['Car'][key][metric]['R40']) > 99.9, (key, metric, results['Car'][key][metric])
    assert (tmp_path / 'eval' / 'results.txt').exists() and (tmp_path / 'eval' / 'results.json').exists()


def test_evaluate_script_pred_dir(eval_root, tmp_path):
    """tools/evaluate.py --pred_dir: labels evaluated against themselves give 100 AP."""
    config = small_config(eval_root)
    pred_dir = tmp_path / 'preds'
    pred_dir.mkdir()
    for name in os.listdir(os.path.join(eval_root, 'training', 'label_2')):
        lines = open(os.path.join(eval_root, 'training', 'label_2', name)).read().splitlines()
        (pred_dir / name).write_text(''.join(f'{line} 0.9\n' for line in lines))

    config_path = tmp_path / 'cfg.yaml'
    config_path.write_text(yaml.safe_dump(config))
    result = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, 'tools', 'evaluate.py'), '--config', str(config_path),
         '--pred_dir', str(pred_dir), '--output_dir', str(tmp_path / 'out')],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    assert '3d   AP: 100.0000, 100.0000, 100.0000' in result.stdout
