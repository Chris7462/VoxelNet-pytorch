import numpy as np

from voxelnet_torch.datasets import Voxelizer

RANGE = [0.0, -40.0, -3.0, 70.4, 40.0, 1.0]
VOXEL = [0.2, 0.2, 0.4]


def reference_voxelize(lidar, T=35):
    """The original per-voxel loop from data/kitti.py (KittiDataset.preprocess), minus the shuffle."""
    voxel_coords = ((lidar[:, :3] - np.array([RANGE[0], RANGE[1], RANGE[2]])) / (
        VOXEL[0], VOXEL[1], VOXEL[2])).astype(np.int32)
    voxel_coords = voxel_coords[:, [2, 1, 0]]
    voxel_coords, inv_ind, voxel_counts = np.unique(voxel_coords, axis=0,
                                                    return_inverse=True, return_counts=True)
    inv_ind = inv_ind.reshape(-1)
    voxel_features = []
    for i in range(len(voxel_coords)):
        voxel = np.zeros((T, 7), dtype=np.float32)
        pts = lidar[inv_ind == i]
        if voxel_counts[i] > T:
            pts = pts[:T, :]
            voxel_counts[i] = T
        voxel[:pts.shape[0], :] = np.concatenate((pts, pts[:, :3] - np.mean(pts[:, :3], 0)), axis=1)
        voxel_features.append(voxel)
    return np.array(voxel_features), voxel_coords, voxel_counts


def random_cloud(rng, n=20000):
    pts = np.column_stack([
        rng.uniform(0, 70.39, n), rng.uniform(-39.99, 39.99, n),
        rng.uniform(-2.99, 0.99, n), rng.uniform(0, 1, n),
    ]).astype(np.float32)
    # A dense cluster so that some voxels exceed T points
    cluster = np.column_stack([
        rng.uniform(10.0, 10.19, 500), rng.uniform(5.0, 5.19, 500),
        rng.uniform(-1.0, -0.61, 500), rng.uniform(0, 1, 500),
    ]).astype(np.float32)
    return np.concatenate([pts, cluster])


def test_matches_original_implementation():
    rng = np.random.default_rng(0)
    points = random_cloud(rng)

    features, num_points, coords = Voxelizer(RANGE, VOXEL, 35)(points)
    ref_features, ref_coords, ref_counts = reference_voxelize(points)

    assert features.shape == ref_features.shape
    np.testing.assert_array_equal(coords, ref_coords)
    np.testing.assert_array_equal(num_points, ref_counts)
    np.testing.assert_allclose(features, ref_features, atol=1e-5)
    assert num_points.max() == 35


def test_empty_cloud():
    features, num_points, coords = Voxelizer(RANGE, VOXEL, 35)(np.zeros((0, 4), np.float32))
    assert features.shape == (0, 35, 7) and num_points.shape == (0,) and coords.shape == (0, 3)


def test_coords_within_grid():
    rng = np.random.default_rng(1)
    voxelizer = Voxelizer(RANGE, VOXEL, 35)
    _, _, coords = voxelizer(random_cloud(rng))
    nx, ny, nz = voxelizer.grid_size
    assert (coords >= 0).all()
    assert coords[:, 0].max() < nz and coords[:, 1].max() < ny and coords[:, 2].max() < nx
