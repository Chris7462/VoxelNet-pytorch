import numpy as np

from ..utils.config import compute_grid_size


class Voxelizer:
    """
    Group points into voxels and build the per-point VoxelNet input features.

    Each voxel keeps at most `max_points` points; each kept point is augmented with
    its offset to the centroid of the kept points, giving 7 features:
    [x, y, z, r, x - cx, y - cy, z - cz].

    Fully vectorized (sort-based grouping), replacing the original per-voxel Python loop.

    Args:
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        voxel_size: [vx, vy, vz]
        max_points: Maximum number of points per voxel (T)
    """

    NUM_FEATURES = 7

    def __init__(
        self,
        point_cloud_range: list[float],
        voxel_size: list[float],
        max_points: int,
    ) -> None:
        self.point_cloud_range = np.asarray(point_cloud_range, dtype=np.float64)
        self.voxel_size = np.asarray(voxel_size, dtype=np.float64)
        self.max_points = max_points
        self.grid_size = compute_grid_size(point_cloud_range, voxel_size)  # (nx, ny, nz)

    def __call__(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Voxelize a point cloud.

        Points must already be filtered to the point cloud range. When a voxel holds more
        than `max_points` points, the first ones (in input order) are kept, so shuffle the
        points beforehand for random sampling.

        Args:
            points: (P, 4) points [x, y, z, r]

        Returns:
            features: (V, T, 7) float32 point features, zero-padded
            num_points: (V,) int32 number of valid points in each voxel
            coords: (V, 3) int32 voxel indices as (z, y, x)
        """
        nx, ny, nz = self.grid_size
        T = self.max_points

        if len(points) == 0:
            return (np.zeros((0, T, self.NUM_FEATURES), dtype=np.float32),
                    np.zeros((0,), dtype=np.int32),
                    np.zeros((0, 3), dtype=np.int32))

        # Voxel index of every point, as (x, y, z)
        xyz_idx = np.floor((points[:, :3] - self.point_cloud_range[:3]) / self.voxel_size).astype(np.int64)
        xyz_idx = np.clip(xyz_idx, 0, np.array([nx - 1, ny - 1, nz - 1]))
        linear = (xyz_idx[:, 2] * ny + xyz_idx[:, 1]) * nx + xyz_idx[:, 0]

        # Group points by voxel; the stable sort keeps the input order inside each voxel
        unique_linear, inverse, counts = np.unique(linear, return_inverse=True, return_counts=True)
        inverse = inverse.reshape(-1)
        order = np.argsort(inverse, kind='stable')
        voxel_of_point = inverse[order]
        starts = np.cumsum(counts) - counts
        slot = np.arange(len(points)) - starts[voxel_of_point]

        keep = slot < T
        voxel_of_point, slot = voxel_of_point[keep], slot[keep]
        kept_points = points[order[keep]].astype(np.float32)

        num_voxels = len(unique_linear)
        num_points = np.minimum(counts, T).astype(np.int32)

        # Centroid of the kept points of each voxel
        sums = np.zeros((num_voxels, 3), dtype=np.float64)
        np.add.at(sums, voxel_of_point, kept_points[:, :3])
        centroids = sums / num_points[:, None]

        features = np.zeros((num_voxels, T, self.NUM_FEATURES), dtype=np.float32)
        features[voxel_of_point, slot, :4] = kept_points[:, :4]
        features[voxel_of_point, slot, 4:] = kept_points[:, :3] - centroids[voxel_of_point]

        coords = np.stack([
            unique_linear // (ny * nx),
            (unique_linear // nx) % ny,
            unique_linear % nx,
        ], axis=1).astype(np.int32)

        return features, num_points, coords
