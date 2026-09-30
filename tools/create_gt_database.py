"""
Build the ground-truth database used by GT sampling augmentation (SECOND-style).

For every labeled object of the training split, the LiDAR points inside its box are
extracted from the (FOV-cropped) point cloud. All points go into one array
(`gt_database/points.npy`), and `gt_database/infos.pkl` lists, per object, its class,
box, difficulty and the slice of points that belongs to it.

Only the training split is used, so no validation object leaks into training.

Usage:
    python tools/create_gt_database.py --config configs/voxelnet_kitti_car.yaml
"""

import argparse
import os
import pickle

import numpy as np

from voxelnet_torch.datasets.kitti_io import (
    KITTI_CLASSES,
    annotations_difficulty,
    annotations_to_lidar_boxes,
    read_calib,
    read_label_annotations,
    read_lidar,
)
from voxelnet_torch.utils import load_config, points_in_boxes


def parse_args():
    parser = argparse.ArgumentParser(description='Create the GT database for GT sampling')
    parser.add_argument('--config', type=str, required=True,
                        help='Config file (dataset.root, dataset.lidar_dir)')
    parser.add_argument('--split', type=str, default='train',
                        help='Split to extract objects from (ImageSets/<split>.txt)')
    parser.add_argument('--output_dir', type=str, default='gt_database',
                        help='Output directory, relative to dataset.root')
    parser.add_argument('--classes', type=str, nargs='+', default=list(KITTI_CLASSES),
                        help='Object classes to store')
    return parser.parse_args()


def create_gt_database(root: str, lidar_dir: str, split: str, output_dir: str, classes: list[str]) -> dict:
    """
    Extract the points of every labeled object of `split`.

    Returns:
        Number of stored objects per class
    """
    with open(os.path.join(root, 'ImageSets', f'{split}.txt')) as f:
        frame_ids = [line.strip() for line in f if line.strip()]

    data_dir = os.path.join(root, 'training')
    out_dir = os.path.join(root, output_dir)
    os.makedirs(out_dir, exist_ok=True)

    infos, chunks = [], []
    offset = 0
    for i, frame_id in enumerate(frame_ids):
        calib = read_calib(os.path.join(data_dir, 'calib', f'{frame_id}.txt'))
        annos = read_label_annotations(os.path.join(data_dir, 'label_2', f'{frame_id}.txt'))
        keep = np.isin(annos['name'], classes)
        if not keep.any():
            continue
        annos = {key: value[keep] for key, value in annos.items()}

        points = read_lidar(os.path.join(data_dir, lidar_dir, f'{frame_id}.bin'))
        boxes = annotations_to_lidar_boxes(annos, calib)
        difficulty = annotations_difficulty(annos)
        inside = points_in_boxes(points, boxes)                       # (P, N)

        for j in range(len(boxes)):
            obj_points = points[inside[:, j]].astype(np.float32)
            infos.append({
                'name': str(annos['name'][j]),
                'frame_id': frame_id,
                'box': boxes[j].astype(np.float32),
                'difficulty': int(difficulty[j]),
                'num_points': len(obj_points),
                'offset': offset,
            })
            chunks.append(obj_points)
            offset += len(obj_points)

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(frame_ids)} frames, {len(infos)} objects")

    all_points = np.concatenate(chunks) if chunks else np.zeros((0, 4), dtype=np.float32)
    np.save(os.path.join(out_dir, 'points.npy'), all_points)
    with open(os.path.join(out_dir, 'infos.pkl'), 'wb') as f:
        pickle.dump({'split': split, 'lidar_dir': lidar_dir, 'infos': infos}, f)

    counts = {}
    for info in infos:
        counts[info['name']] = counts.get(info['name'], 0) + 1
    return counts


def main():
    args = parse_args()
    config = load_config(args.config)
    root = config['dataset']['root']
    lidar_dir = config['dataset'].get('lidar_dir', 'crop')

    print(f"Creating GT database from {root} ({args.split} split, points from '{lidar_dir}')")
    counts = create_gt_database(root, lidar_dir, args.split, args.output_dir, args.classes)
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count} objects")
    print(f"Saved to {os.path.join(root, args.output_dir)}")


if __name__ == '__main__':
    main()
