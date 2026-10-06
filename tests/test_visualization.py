"""Bird's-eye-view and camera-image visualization of boxes."""

import os
import subprocess
import sys

import cv2
import numpy as np
import yaml

from voxelnet_torch.datasets.kitti_io import lidar_boxes_to_annotations, read_calib
from voxelnet_torch.engine.evaluator import visualize_predictions
from voxelnet_torch.utils import draw_bev, draw_boxes_on_image, visualize_detections
from voxelnet_torch.utils.visualization import (
    BOX_EDGES, GT_COLOR, PRED_COLOR, bev_pixels, project_boxes_to_image,
)

from conftest import CALIB_TEXT, REPO_ROOT, small_config
from test_evaluator import IMAGE_SHAPE, eval_root  # noqa: F401  (fixture)

RANGE = [0.0, -12.8, -3.0, 25.6, 12.8, 1.0]


def calibration(tmp_path) -> dict:
    calib_file = tmp_path / 'calib.txt'
    calib_file.write_text(CALIB_TEXT)
    return read_calib(str(calib_file))


def color_mask(image: np.ndarray, color: tuple) -> np.ndarray:
    """Pixels drawn in a pure color (lines are anti-aliased, so they blend with the background)."""
    on = np.array(color) > 0
    return image[..., on].min(axis=2).astype(int) - image[..., ~on].max(axis=2) > 40


def test_bev_pixels_orientation():
    # Forward (x) is up, left (y) is left
    np.testing.assert_allclose(bev_pixels([25.6, 12.8], RANGE, 0.1), [0, 0])
    np.testing.assert_allclose(bev_pixels([0.0, -12.8], RANGE, 0.1), [256, 256])
    np.testing.assert_allclose(bev_pixels([[10.0, 0.0]], RANGE, 0.2), [[64, 78]])


def test_draw_bev():
    points = np.array([[10.05, 5.05, -1.0, 0.5], [20.05, -3.05, 0.5, 0.5], [100.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    gt = np.array([[8.0, -6.0, -1.7, 1.5, 1.6, 4.0, 0.0]])
    pred = np.array([[18.0, 6.0, -1.7, 1.5, 1.6, 4.0, np.pi / 2]])

    image = draw_bev(points, RANGE, gt, pred, scores=np.array([0.87]))
    assert image.shape == (256, 256, 3) and image.dtype == np.uint8

    # Points: (row, col) = ((x_max - x) / res, (y_max - y) / res); higher points are brighter
    low, high = image[155, 77], image[55, 158]
    assert low[0] == low[1] == low[2] and 0 < low[0] < high[0]

    # Without boxes only the two points inside the range are drawn
    assert (draw_bev(points, RANGE).sum(axis=2) > 0).sum() == 2
    assert draw_bev(np.zeros((0, 4)), RANGE).sum() == 0

    # GT footprint: x in [6, 10], y in [-6.8, -5.2] -> rows 156..196, cols 180..196
    gt_rows, gt_cols = np.nonzero(color_mask(image, GT_COLOR))
    assert len(gt_rows) > 100
    assert 154 <= gt_rows.min() and gt_rows.max() <= 198 and 178 <= gt_cols.min() and gt_cols.max() <= 198
    assert color_mask(image, GT_COLOR)[176, 180] and color_mask(image, GT_COLOR)[156, 188]   # side and front edges
    assert color_mask(image, GT_COLOR)[166, 188]                                             # heading line (front half)
    assert not color_mask(image, GT_COLOR)[186, 188]                                         # ... not the rear half

    # Prediction rotated by 90 deg (front towards +y = left): x in [17.2, 18.8], y in [4, 8]
    # -> rows 68..84, cols 48..88, score text to its right
    pred_mask = color_mask(image, PRED_COLOR)
    assert pred_mask[68, 68] and pred_mask[76, 48] and pred_mask[76, 58] and not pred_mask[76, 78]
    assert pred_mask[:, 92:].any()                                                           # score label


def test_project_boxes_matches_2d_box(tmp_path):
    calib = calibration(tmp_path)
    boxes = np.array([
        [15.0, 2.0, -1.7, 1.5, 1.6, 3.9, 0.3],      # in front of the camera
        [-15.0, 0.0, -1.7, 1.5, 1.6, 3.9, 0.0],     # behind it
        [0.5, 0.0, -1.7, 1.5, 1.6, 3.9, 0.0],       # around the camera: partly behind
    ])
    segments = project_boxes_to_image(boxes, calib)
    assert len(segments) == 3

    assert segments[0].shape == (len(BOX_EDGES), 2, 2)
    annos = lidar_boxes_to_annotations(boxes[:1], np.ones(1), calib, IMAGE_SHAPE)
    points = segments[0].reshape(-1, 2)
    np.testing.assert_allclose(np.concatenate([points.min(axis=0), points.max(axis=0)]), annos['bbox'][0], atol=1e-6)

    assert segments[1].shape == (0, 2, 2)
    assert 0 < len(segments[2]) < len(BOX_EDGES) and np.isfinite(segments[2]).all()
    assert project_boxes_to_image(np.zeros((0, 7)), calib) == []


def test_draw_boxes_on_image(tmp_path):
    calib = calibration(tmp_path)
    boxes = np.array([[15.0, 2.0, -1.7, 1.5, 1.6, 3.9, 0.3], [-15.0, 0.0, -1.7, 1.5, 1.6, 3.9, 0.0],
                      [0.5, 0.0, -1.7, 1.5, 1.6, 3.9, 0.0], [2.0, 30.0, -1.7, 1.5, 1.6, 3.9, 0.0]])
    image = np.zeros((*IMAGE_SHAPE, 3), dtype=np.uint8)

    assert draw_boxes_on_image(image, boxes[:1], calib, PRED_COLOR, thickness=1) is image
    rows, cols = np.nonzero(color_mask(image, PRED_COLOR))
    x1, y1, x2, y2 = lidar_boxes_to_annotations(boxes[:1], np.ones(1), calib, IMAGE_SHAPE)['bbox'][0]
    assert len(rows) > 200
    assert x1 - 2 <= cols.min() and cols.max() <= x2 + 2 and y1 - 2 <= rows.min() and rows.max() <= y2 + 2
    assert cols.min() <= x1 + 2 and cols.max() >= x2 - 2 and rows.min() <= y1 + 2 and rows.max() >= y2 - 2

    # Boxes behind / around / far beside the camera, with scores: no error, image keeps its shape
    before = image.copy()
    draw_boxes_on_image(image, boxes[1:2], calib, GT_COLOR, scores=np.array([0.5]))
    np.testing.assert_array_equal(image, before)                    # fully behind the camera: nothing drawn
    draw_boxes_on_image(image, boxes, calib, GT_COLOR, scores=np.array([0.9, 0.5, 0.4, 0.3]))
    assert image.shape == (*IMAGE_SHAPE, 3) and color_mask(image, GT_COLOR).sum() > 200


def test_visualize_detections(tmp_path):
    calib = calibration(tmp_path)
    points = np.random.default_rng(0).uniform([0, -12, -2, 0], [25, 12, 0, 1], size=(500, 4)).astype(np.float32)
    gt = np.array([[15.0, 2.0, -1.7, 1.5, 1.6, 3.9, 0.3]], dtype=np.float32)
    camera = np.full((*IMAGE_SHAPE, 3), 60, dtype=np.uint8)

    picture = visualize_detections(points, RANGE, gt, gt + 0.2, np.array([0.7]), image=camera, calib=calib, title='000001')
    assert picture.shape == (IMAGE_SHAPE[0] + 256, IMAGE_SHAPE[1], 3)
    assert (camera == 60).all()                                     # input image is not modified
    for part in (picture[:IMAGE_SHAPE[0]], picture[IMAGE_SHAPE[0]:]):
        assert color_mask(part, GT_COLOR).any() and color_mask(part, PRED_COLOR).any()

    # No camera image: bird's-eye view only; no boxes at all is fine too
    assert visualize_detections(points, RANGE, gt, None).shape == (256, 256, 3)
    assert visualize_detections(points, RANGE).shape == (256, 256, 3)


def write_predictions(root: str, pred_dir, score: float) -> None:
    """Predictions = the labels, with a score column."""
    pred_dir.mkdir()
    label_dir = os.path.join(root, 'training', 'label_2')
    for name in os.listdir(label_dir):
        lines = open(os.path.join(label_dir, name)).read().splitlines()
        (pred_dir / name).write_text(''.join(f'{line} {score}\n' for line in lines))


def test_visualize_predictions(eval_root, tmp_path):  # noqa: F811
    config = small_config(eval_root)
    write_predictions(eval_root, tmp_path / 'preds', 0.9)
    os.remove(tmp_path / 'preds' / '000001.txt')                          # frame without predictions
    os.remove(os.path.join(eval_root, 'training', 'image_2', '000002.png'))   # frame without camera image
    frame_ids = [f'{i:06d}' for i in range(12)]

    out = visualize_predictions(config, tmp_path / 'preds', frame_ids, tmp_path / 'vis', num_visualize=3)
    assert sorted(os.listdir(out)) == ['000000.png', '000001.png', '000002.png']

    with_pred = cv2.imread(str(out / '000000.png'))
    without_pred = cv2.imread(str(out / '000001.png'))
    assert with_pred.shape == (IMAGE_SHAPE[0] + 256, IMAGE_SHAPE[1], 3)
    assert cv2.imread(str(out / '000002.png')).shape == (256, 256, 3)
    assert color_mask(with_pred, PRED_COLOR).sum() > color_mask(without_pred, PRED_COLOR).sum() + 500
    assert color_mask(without_pred, GT_COLOR).sum() > 500

    # Predictions below the score threshold are not drawn; num_visualize=None draws every frame
    out = visualize_predictions(config, tmp_path / 'preds', frame_ids, tmp_path / 'vis_high', None, score_threshold=0.95)
    assert len(os.listdir(out)) == 12
    hidden = cv2.imread(str(out / '000000.png'))
    assert color_mask(hidden, PRED_COLOR).sum() < color_mask(with_pred, PRED_COLOR).sum() - 500


def test_evaluate_script_visualize(eval_root, tmp_path):  # noqa: F811
    config = small_config(eval_root)
    write_predictions(eval_root, tmp_path / 'preds', 0.9)
    config_path = tmp_path / 'cfg.yaml'
    config_path.write_text(yaml.safe_dump(config))

    result = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, 'tools', 'evaluate.py'), '--config', str(config_path),
         '--pred_dir', str(tmp_path / 'preds'), '--output_dir', str(tmp_path / 'out'),
         '--visualize', '--num_visualize', '2'],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    assert sorted(os.listdir(tmp_path / 'out' / 'visualizations')) == ['000000.png', '000001.png']
