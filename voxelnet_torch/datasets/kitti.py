import os

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.anchors import build_anchors
from ..utils.box_ops import boxes_to_corners_3d
from .augmentation import Augmentor
from .gt_sampler import GTSampler
from .kitti_io import KITTI_CLASSES, read_calib, read_label, read_lidar
from .target_assigner import TargetAssigner
from .voxelizer import Voxelizer


class KITTI(Dataset):
    """
    KITTI 3D object detection dataset for VoxelNet.

    Expected layout:
        root/
        ├── ImageSets/{train,val}.txt   # frame ids, one per line
        └── training/
            ├── calib/
            ├── label_2/
            └── <lidar_dir>/            # 'crop' (see tools/crop_kitti.py) or 'velodyne'

    Args:
        config: Full configuration dictionary
        image_set: Split name, e.g. 'train' or 'val' (reads ImageSets/<image_set>.txt)
        training: Enable augmentation and point shuffling
        assign_targets: Compute the RPN training targets (not needed for inference)
    """

    def __init__(
        self,
        config: dict,
        image_set: str,
        training: bool = False,
        assign_targets: bool = True,
    ) -> None:
        super().__init__()

        data_cfg = config['dataset']
        self.root = data_cfg['root']
        self.image_set = image_set
        self.training = training
        self.assign_targets = assign_targets
        self.classes = data_cfg['classes']

        data_dir = os.path.join(self.root, 'training')
        self.lidar_dir = os.path.join(data_dir, data_cfg.get('lidar_dir', 'crop'))
        self.calib_dir = os.path.join(data_dir, 'calib')
        self.label_dir = os.path.join(data_dir, 'label_2')

        split_file = os.path.join(self.root, 'ImageSets', f'{image_set}.txt')
        with open(split_file) as f:
            self.frame_ids = [line.strip() for line in f if line.strip()]

        self.point_cloud_range = np.asarray(data_cfg['point_cloud_range'], dtype=np.float32)
        self.voxelizer = Voxelizer(
            data_cfg['point_cloud_range'],
            data_cfg['voxel_size'],
            data_cfg['max_points_per_voxel'],
        )

        anchor_cfg = config['anchor']
        self.target_assigner = TargetAssigner(
            build_anchors(config),
            pos_iou=anchor_cfg['pos_iou'],
            neg_iou=anchor_cfg['neg_iou'],
        )

        aug_cfg = config.get('augmentation', {})
        aug_enabled = training and aug_cfg.get('enabled', False)
        self.augmentor = Augmentor(aug_cfg) if aug_enabled else None

        # GT sampling: paste objects from other training frames (applied before the other augmentations)
        sampling_cfg = aug_cfg.get('gt_sampling', {})
        self.gt_sampler = GTSampler(self.root, sampling_cfg) if aug_enabled and sampling_cfg.get('enabled', False) else None
        self.other_classes = [c for c in KITTI_CLASSES if c not in self.classes]

        self._rng = None
        self._rng_seed = None

    def _get_rng(self) -> np.random.Generator:
        """
        Random generator seeded from torch's per-process seed.

        DataLoader workers get a different torch seed per worker (and per epoch,
        unless workers are persistent), so augmentation differs across workers
        while staying reproducible under `set_seed`.
        """
        seed = torch.initial_seed()
        if self._rng is None or self._rng_seed != seed:
            self._rng = np.random.default_rng(seed)
            self._rng_seed = seed
        return self._rng

    def _filter_by_range(
        self,
        points: np.ndarray,
        boxes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Keep points inside the range, and boxes with at least one corner inside it."""
        low, high = self.point_cloud_range[:3], self.point_cloud_range[3:]

        point_mask = np.all((points[:, :3] >= low) & (points[:, :3] < high), axis=1)

        corners = boxes_to_corners_3d(boxes)                                    # (N, 8, 3)
        corner_inside = np.all((corners >= low) & (corners < high), axis=2)     # (N, 8)
        box_mask = corner_inside.any(axis=1)

        return points[point_mask], boxes[box_mask]

    def __getitem__(self, idx: int) -> dict:
        frame_id = self.frame_ids[idx]

        calib = read_calib(os.path.join(self.calib_dir, f'{frame_id}.txt'))
        points = read_lidar(os.path.join(self.lidar_dir, f'{frame_id}.bin'))
        gt_boxes = read_label(os.path.join(self.label_dir, f'{frame_id}.txt'), calib, self.classes)

        if self.training:
            rng = self._get_rng()
            if self.gt_sampler is not None:
                label_file = os.path.join(self.label_dir, f'{frame_id}.txt')
                other_boxes = read_label(label_file, calib, self.other_classes)
                points, gt_boxes = self.gt_sampler(points, gt_boxes, other_boxes, rng)
            if self.augmentor is not None:
                points, gt_boxes = self.augmentor(points, gt_boxes, rng)

        points, gt_boxes = self._filter_by_range(points, gt_boxes)

        if self.training:
            # Random subset of points in voxels holding more than T points
            points = points[rng.permutation(len(points))]

        voxels, num_points, coords = self.voxelizer(points)

        sample = {
            'voxels': voxels,
            'num_points': num_points,
            'coords': coords,
            'gt_boxes': gt_boxes.astype(np.float32),
            'frame_id': frame_id,
        }

        if self.assign_targets:
            pos_equal_one, neg_equal_one, targets = self.target_assigner(gt_boxes)
            sample.update(pos_equal_one=pos_equal_one, neg_equal_one=neg_equal_one, targets=targets)

        return sample

    def __len__(self) -> int:
        return len(self.frame_ids)

    @staticmethod
    def collate(batch: list[dict]) -> dict:
        """
        Custom collate function for DataLoader.

        Voxels of all samples are concatenated; `coords` gains a leading batch index column.

        Args:
            batch: List of samples from __getitem__

        Returns:
            Dictionary with batched tensors
        """
        coords = [
            np.pad(b['coords'], ((0, 0), (1, 0)), mode='constant', constant_values=i)
            for i, b in enumerate(batch)
        ]

        collated = {
            'voxels': torch.from_numpy(np.concatenate([b['voxels'] for b in batch])),
            'num_points': torch.from_numpy(np.concatenate([b['num_points'] for b in batch])).long(),
            'coords': torch.from_numpy(np.concatenate(coords)).long(),
            'gt_boxes': [torch.from_numpy(b['gt_boxes']) for b in batch],
            'frame_id': [b['frame_id'] for b in batch],
            'batch_size': len(batch),
        }

        for key in ('pos_equal_one', 'neg_equal_one', 'targets'):
            if key in batch[0]:
                collated[key] = torch.from_numpy(np.stack([b[key] for b in batch]))

        return collated
