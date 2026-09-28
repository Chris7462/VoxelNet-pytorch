import yaml


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def compute_grid_size(
    point_cloud_range: list[float],
    voxel_size: list[float],
) -> tuple[int, int, int]:
    """
    Compute the voxel grid size from the point cloud range and voxel size.

    Uses rounding instead of ceil so that floating point noise
    (e.g. 70.4 / 0.2 = 351.99999999999994) does not change the result.

    Args:
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        voxel_size: [vx, vy, vz]

    Returns:
        (nx, ny, nz) number of voxels along x, y, z
    """
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    vx, vy, vz = voxel_size

    nx = int(round((x_max - x_min) / vx))
    ny = int(round((y_max - y_min) / vy))
    nz = int(round((z_max - z_min) / vz))

    return nx, ny, nz
