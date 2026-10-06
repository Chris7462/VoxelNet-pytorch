import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from ..datasets.kitti_io import (
    KITTI_CLASSES,
    annotations_to_lidar_boxes,
    lidar_boxes_to_annotations,
    read_calib,
    read_image_shape,
    read_label_annotations,
    read_lidar,
    write_annotations,
)
from ..utils import build_anchors, postprocess, visualize_detections
from ..utils.kitti_eval import evaluate, format_results


class Evaluator:
    """
    Evaluator for VoxelNet on KITTI.

    Runs inference on a labeled split, writes the detections in the KITTI label format
    (one `<frame>.txt` per frame, usable with the official devkit), and computes
    the KITTI 2D / BEV / 3D AP.

    Args:
        model: VoxelNet model
        data_loader: Data loader over a KITTI split (dataset built with assign_targets=False)
        config: Configuration dictionary
        device: Device to run on
        output_dir: Directory for predictions and results
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader,
        config: dict,
        device: torch.device,
        output_dir: str | Path,
    ) -> None:
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.device = torch.device(device)

        self.output_dir = Path(output_dir)
        self.pred_dir = self.output_dir / 'predictions'
        self.pred_dir.mkdir(parents=True, exist_ok=True)

        eval_cfg = config['evaluation']
        self.score_threshold = eval_cfg['score_threshold']
        self.nms_iou_threshold = eval_cfg['nms_iou_threshold']
        self.max_detections = eval_cfg['max_detections']
        self.nms_type = eval_cfg.get('nms_type', 'standup')
        self.pre_nms_top_k = eval_cfg.get('pre_nms_top_k', None)
        self.use_amp = bool(config['train'].get('amp', False)) and self.device.type == 'cuda'

        data_dir = os.path.join(config['dataset']['root'], 'training')
        self.calib_dir = os.path.join(data_dir, 'calib')
        self.image_dir = os.path.join(data_dir, 'image_2')
        self.label_dir = os.path.join(data_dir, 'label_2')

        self.anchors = torch.from_numpy(build_anchors(config)).to(self.device)

    @torch.no_grad()
    def predict(self) -> list[str]:
        """
        Run inference and write one KITTI-format prediction file per frame.

        Returns:
            Frame ids in dataset order
        """
        self.model.eval()
        frame_ids = []

        num_batches = len(self.data_loader)
        print(f"Running inference... ({num_batches} batches)")

        for batch_idx, sample in enumerate(self.data_loader):
            voxels = sample['voxels'].to(self.device, non_blocking=True)
            num_points = sample['num_points'].to(self.device, non_blocking=True)
            coords = sample['coords'].to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                psm, rm = self.model(voxels, num_points, coords, sample['batch_size'])

            detections = postprocess(
                psm, rm, self.anchors,
                score_threshold=self.score_threshold,
                nms_iou_threshold=self.nms_iou_threshold,
                max_detections=self.max_detections,
                nms_type=self.nms_type,
                pre_nms_top_k=self.pre_nms_top_k,
            )

            for frame_id, det in zip(sample['frame_id'], detections):
                calib = read_calib(os.path.join(self.calib_dir, f'{frame_id}.txt'))
                image_shape = read_image_shape(os.path.join(self.image_dir, f'{frame_id}.png'))
                annos = lidar_boxes_to_annotations(
                    det['boxes'].cpu().numpy(), det['scores'].cpu().numpy(), calib, image_shape)
                write_annotations(annos, str(self.pred_dir / f'{frame_id}.txt'))
                frame_ids.append(frame_id)

            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {batch_idx + 1}/{num_batches} batches")

        print(f"Predictions saved to: {self.pred_dir}")
        return frame_ids

    def evaluate(self, frame_ids: list[str], classes: tuple[str, ...] = ('Car',)) -> dict:
        """
        Compute KITTI AP from the prediction files against the ground-truth labels.

        Args:
            frame_ids: Frames to evaluate
            classes: Classes to evaluate

        Returns:
            Results dict (see `voxelnet_torch.utils.kitti_eval.evaluate`)
        """
        return evaluate_predictions(self.pred_dir, self.label_dir, frame_ids, classes, self.output_dir)

    def visualize(
        self,
        frame_ids: list[str],
        num_visualize: int | None = 20,
        score_threshold: float | None = None,
    ) -> Path:
        """
        Save pictures of the written predictions (see `visualize_predictions`).

        Args:
            frame_ids: Frames to draw
            num_visualize: Draw only the first N frames (None: all)
            score_threshold: Minimum score of a drawn prediction (default: evaluation.vis_score_threshold, 0.95)

        Returns:
            Directory with the pictures
        """
        return visualize_predictions(self.config, self.pred_dir, frame_ids, self.output_dir / 'visualizations',
                                     num_visualize, score_threshold)


def evaluate_predictions(
    pred_dir: str | Path,
    label_dir: str | Path,
    frame_ids: list[str],
    classes: tuple[str, ...] = ('Car',),
    output_dir: str | Path | None = None,
) -> dict:
    """
    Evaluate KITTI-format prediction files (with a score column) against the labels.

    Frames without a prediction file count as frames without detections.
    Results are printed and, if `output_dir` is given, saved as results.txt / results.json.
    """
    print(f"Evaluating {len(frame_ids)} frames...")
    gt_annos, dt_annos = [], []
    for frame_id in frame_ids:
        gt_annos.append(read_label_annotations(os.path.join(label_dir, f'{frame_id}.txt')))

        pred_file = os.path.join(pred_dir, f'{frame_id}.txt')
        dt = read_label_annotations(pred_file) if os.path.exists(pred_file) else read_label_annotations(os.devnull)
        if 'score' not in dt:
            if len(dt['name']) > 0:
                raise ValueError(f"Prediction file without score column: {pred_file}")
            dt['score'] = np.zeros(0)
        dt_annos.append(dt)

    results = evaluate(gt_annos, dt_annos, classes)
    table = format_results(results)
    print(table)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'results.txt').write_text(table + '\n')
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_dir / 'results.txt'}")

    return results


def visualize_predictions(
    config: dict,
    pred_dir: str | Path,
    frame_ids: list[str],
    output_dir: str | Path,
    num_visualize: int | None = 20,
    score_threshold: float | None = None,
) -> Path:
    """
    Draw KITTI-format prediction files next to the ground truth, one `<frame>.png` per frame:
    the camera image with the projected 3D boxes on top, the bird's-eye view of the point cloud below.
    Ground truth (the training classes) is green, predictions are red with their score.
    Labeled objects of the other classes are yellow with their class name, and DontCare
    regions are cyan rectangles in the camera image.

    The default score threshold is high because the scores of this model saturate:
    true cars mostly score close to 1, and many false alarms score between 0.5 and 0.95.

    Frames without a prediction file are drawn without predictions; frames without
    a camera image are drawn as bird's-eye view only.

    Args:
        config: Configuration dictionary
        pred_dir: Directory with the prediction files
        frame_ids: Frames to draw
        output_dir: Directory for the pictures
        num_visualize: Draw only the first N frames (None: all)
        score_threshold: Minimum score of a drawn prediction (default: evaluation.vis_score_threshold, 0.95)

    Returns:
        `output_dir`
    """
    data_cfg = config['dataset']
    data_dir = os.path.join(data_cfg['root'], 'training')
    lidar_dir = os.path.join(data_dir, data_cfg.get('lidar_dir', 'crop'))
    if score_threshold is None:
        score_threshold = config.get('evaluation', {}).get('vis_score_threshold', 0.95)
    classes = list(data_cfg['classes'])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_ids = frame_ids if num_visualize is None else frame_ids[:num_visualize]
    print(f"Visualizing {len(frame_ids)} frames (predictions with score > {score_threshold})...")

    for frame_id in frame_ids:
        calib = read_calib(os.path.join(data_dir, 'calib', f'{frame_id}.txt'))
        points = read_lidar(os.path.join(lidar_dir, f'{frame_id}.bin'))
        image = cv2.imread(os.path.join(data_dir, 'image_2', f'{frame_id}.png'))      # None if missing

        labels = read_label_annotations(os.path.join(data_dir, 'label_2', f'{frame_id}.txt'))
        is_gt = np.isin(labels['name'], classes)
        is_other = np.isin(labels['name'], KITTI_CLASSES) & ~is_gt
        gt_boxes = annotations_to_lidar_boxes({k: v[is_gt] for k, v in labels.items()}, calib)
        other_boxes = annotations_to_lidar_boxes({k: v[is_other] for k, v in labels.items()}, calib)

        pred_boxes, scores = np.zeros((0, 7)), np.zeros(0)
        pred_file = os.path.join(pred_dir, f'{frame_id}.txt')
        if os.path.exists(pred_file):
            annos = read_label_annotations(pred_file)
            pred_boxes = annotations_to_lidar_boxes(annos, calib)
            scores = annos.get('score', np.ones(len(pred_boxes)))
            keep = scores > score_threshold
            pred_boxes, scores = pred_boxes[keep], scores[keep]

        picture = visualize_detections(
            points, data_cfg['point_cloud_range'], gt_boxes, pred_boxes, scores,
            image=image, calib=calib, title=frame_id,
            other_boxes=other_boxes, other_labels=labels['name'][is_other].tolist(),
            dontcare_bboxes=labels['bbox'][labels['name'] == 'DontCare'],
        )
        cv2.imwrite(str(output_dir / f'{frame_id}.png'), picture)

    print(f"Visualizations saved to: {output_dir}")
    return output_dir
