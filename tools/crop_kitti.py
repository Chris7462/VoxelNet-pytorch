"""
Crop KITTI point clouds to the left color camera's field of view.

VoxelNet is trained and evaluated on the points that project into image_2.
For every frame this keeps the points that lie in front of the camera and project
inside the image, and writes them to <split>/crop/<frame>.bin.

Usage:
    python tools/crop_kitti.py --root data/KITTI
    python tools/crop_kitti.py --root data/KITTI --splits training
"""

import argparse
import os

import cv2
import numpy as np

from voxelnet_torch.datasets.kitti_io import read_calib, read_lidar, velo_to_image


def parse_args():
    parser = argparse.ArgumentParser(description='Crop KITTI point clouds to the camera FOV')
    parser.add_argument('--root', type=str, default='data/KITTI',
                        help='KITTI object detection root (containing training/ and testing/)')
    parser.add_argument('--splits', type=str, nargs='+', default=['training', 'testing'],
                        help='Sub-directories to process')
    parser.add_argument('--output_dir', type=str, default='crop',
                        help='Output directory name inside each split')
    return parser.parse_args()


def crop_to_fov(points: np.ndarray, calib: dict, image_shape: tuple[int, int]) -> np.ndarray:
    """
    Keep the points that project inside the image and lie in front of the camera.

    Args:
        points: (P, 4) LiDAR points
        calib: Calibration dict from `read_calib`
        image_shape: (height, width) of image_2

    Returns:
        (P', 4) cropped points
    """
    height, width = image_shape
    uv, depth = velo_to_image(points[:, :3].astype(np.float64), calib)

    mask = (
        (depth > 0)
        & (uv[:, 0] >= 0) & (uv[:, 0] < width)
        & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )
    return points[mask]


def main():
    args = parse_args()

    for split in args.splits:
        split_dir = os.path.join(args.root, split)
        velodyne_dir = os.path.join(split_dir, 'velodyne')
        if not os.path.isdir(velodyne_dir):
            print(f"Skipping {split}: {velodyne_dir} not found")
            continue

        output_dir = os.path.join(split_dir, args.output_dir)
        os.makedirs(output_dir, exist_ok=True)

        frame_ids = sorted(f[:-4] for f in os.listdir(velodyne_dir) if f.endswith('.bin'))
        print(f"{split}: cropping {len(frame_ids)} frames -> {output_dir}")

        for i, frame_id in enumerate(frame_ids):
            points = read_lidar(os.path.join(velodyne_dir, f'{frame_id}.bin'))
            calib = read_calib(os.path.join(split_dir, 'calib', f'{frame_id}.txt'))
            image = cv2.imread(os.path.join(split_dir, 'image_2', f'{frame_id}.png'))
            if image is None:
                raise FileNotFoundError(f"Missing image for frame {frame_id} in {split}")

            cropped = crop_to_fov(points, calib, image.shape[:2])
            cropped.astype(np.float32).tofile(os.path.join(output_dir, f'{frame_id}.bin'))

            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(frame_ids)}")

    print("Done.")


if __name__ == '__main__':
    main()
