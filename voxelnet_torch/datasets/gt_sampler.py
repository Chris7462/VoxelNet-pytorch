"""
GT sampling augmentation (as introduced by SECOND): paste objects, with their LiDAR points,
from other training frames into the current scene.
"""

import os
import pickle

import numpy as np

from ..utils.box_ops import (
    boxes_to_standup_bev,
    iou_2d,
    points_in_boxes,
    rotated_bev_iou,
)


class GTSampler:
    """
    Paste database objects into a scene.

    For each class in `sample_groups`, objects are drawn until the scene contains the target
    number of objects of the training classes. Drawn objects stay at the position they had in
    their own frame (the camera / LiDAR geometry is the same for all KITTI frames), and a drawn
    object is rejected if its bird's-eye-view footprint overlaps any object already in the scene
    (including objects of other classes) or another accepted object. Scene points inside the
    accepted boxes are removed before the object points are added.

    Args:
        root: KITTI root directory
        config: The `augmentation.gt_sampling` section of the config
    """

    def __init__(self, root: str, config: dict) -> None:
        db_dir = os.path.join(root, config.get('database_dir', 'gt_database'))
        with open(os.path.join(db_dir, 'infos.pkl'), 'rb') as f:
            infos = pickle.load(f)['infos']

        # Memory-mapped: shared by all DataLoader workers through the page cache
        self.points = np.load(os.path.join(db_dir, 'points.npy'), mmap_mode='r')

        self.sample_groups = dict(config['sample_groups'])
        min_points = config.get('min_points', {})
        filter_difficulty = set(config.get('filter_difficulty', [-1]))

        self.pools = {}
        for name in self.sample_groups:
            pool = [
                info for info in infos
                if info['name'] == name
                and info['num_points'] >= min_points.get(name, 5)
                and info['difficulty'] not in filter_difficulty
            ]
            if not pool:
                raise ValueError(f"No '{name}' objects left in the GT database after filtering")
            self.pools[name] = pool

    def pool_size(self, name: str) -> int:
        return len(self.pools[name])

    def __call__(
        self,
        points: np.ndarray,
        gt_boxes: np.ndarray,
        other_boxes: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            points: (P, 4) scene points
            gt_boxes: (N, 7) boxes of the training classes
            other_boxes: (M, 7) boxes of all other labeled objects (only used to avoid collisions)
            rng: Random generator

        Returns:
            (points, gt_boxes) with the sampled objects added
        """
        occupied = np.concatenate([gt_boxes, other_boxes]).reshape(-1, 7).astype(np.float64)
        sampled_boxes, sampled_points = [], []

        for name, target in self.sample_groups.items():
            num_to_sample = target - len(gt_boxes) - len(sampled_boxes)
            if num_to_sample <= 0:
                continue

            pool = self.pools[name]
            idx = rng.choice(len(pool), size=min(num_to_sample, len(pool)), replace=False)
            for i in idx:
                info = pool[i]
                box = np.asarray(info['box'], dtype=np.float64)[None]
                if len(occupied) > 0:
                    close = iou_2d(boxes_to_standup_bev(box), boxes_to_standup_bev(occupied))[0] > 0
                    if close.any() and (rotated_bev_iou(box, occupied[close]) > 0).any():
                        continue
                occupied = np.concatenate([occupied, box])
                sampled_boxes.append(box[0])
                start = info['offset']
                sampled_points.append(np.asarray(self.points[start:start + info['num_points']]))

        if not sampled_boxes:
            return points, gt_boxes

        sampled_boxes = np.array(sampled_boxes, dtype=np.float32)
        inside = points_in_boxes(points, sampled_boxes).any(axis=1)
        points = np.concatenate([points[~inside]] + sampled_points).astype(np.float32)
        gt_boxes = np.concatenate([gt_boxes.reshape(-1, 7), sampled_boxes]).astype(np.float32)
        return points, gt_boxes
