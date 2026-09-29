"""
Evaluate VoxelNet on the KITTI validation split.

Runs inference with a checkpoint, writes KITTI-format predictions and computes the
2D / BEV / 3D AP (11 and 40 recall points):
    python tools/evaluate.py --config configs/voxelnet_kitti_car.yaml --checkpoint checkpoints/best.pth

Evaluate existing prediction files only (no model needed):
    python tools/evaluate.py --config configs/voxelnet_kitti_car.yaml --pred_dir outputs/eval/predictions
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from voxelnet_torch.datasets import KITTI
from voxelnet_torch.engine import Evaluator
from voxelnet_torch.engine.evaluator import evaluate_predictions
from voxelnet_torch.model import VoxelNet
from voxelnet_torch.utils import compute_grid_size, load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate VoxelNet on KITTI')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint to evaluate')
    parser.add_argument('--pred_dir', type=str, default=None,
                        help='Evaluate existing KITTI-format predictions instead of running the model')
    parser.add_argument('--split', type=str, default='val',
                        help='Split to evaluate (ImageSets/<split>.txt)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: evaluation.output_dir in the config)')
    args = parser.parse_args()
    if (args.checkpoint is None) == (args.pred_dir is None):
        parser.error('Pass exactly one of --checkpoint or --pred_dir')
    return args


def build_model(config: dict):
    """Build VoxelNet model."""
    data_cfg = config['dataset']
    model_cfg = config['model']
    return VoxelNet(
        grid_size=compute_grid_size(data_cfg['point_cloud_range'], data_cfg['voxel_size']),
        num_anchors=len(config['anchor']['rotations']),
        vfe_channels=model_cfg['vfe_channels'],
        voxel_feature_dim=model_cfg['voxel_feature_dim'],
    )


def main():
    args = parse_args()

    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    output_dir = args.output_dir or config['evaluation']['output_dir']
    root = config['dataset']['root']
    classes = ('Car',)

    if args.pred_dir:
        split_file = os.path.join(root, 'ImageSets', f'{args.split}.txt')
        with open(split_file) as f:
            frame_ids = [line.strip() for line in f if line.strip()]
        evaluate_predictions(args.pred_dir, os.path.join(root, 'training', 'label_2'),
                             frame_ids, classes, output_dir)
        return

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = KITTI(config, image_set=args.split, training=False, assign_targets=False)
    data_loader = DataLoader(
        dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=False,
        num_workers=config['dataloader']['num_workers'],
        collate_fn=dataset.collate,
        pin_memory=True,
    )
    print(f"  {args.split}: {len(dataset)} samples, {len(data_loader)} batches")

    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['net'])
    print(f"Loaded checkpoint: {args.checkpoint} (iteration {checkpoint.get('iteration', '?')})")

    evaluator = Evaluator(model, data_loader, config, device, output_dir)
    frame_ids = evaluator.predict()
    evaluator.evaluate(frame_ids, classes)


if __name__ == '__main__':
    main()
