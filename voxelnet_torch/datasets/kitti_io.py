"""Readers for the KITTI object detection file formats."""

import numpy as np

from ..utils.box_ops import boxes_to_corners_3d, limit_period


def read_lidar(lidar_file: str) -> np.ndarray:
    """
    Read a KITTI velodyne .bin file.

    Returns:
        (P, 4) float32 array of [x, y, z, reflectance]
    """
    return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)


def read_calib(calib_file: str) -> dict:
    """
    Read a KITTI calibration file.

    Returns:
        Dict with
            'P2': (3, 4) left color camera projection matrix
            'R0_rect': (4, 4) rectifying rotation (homogeneous)
            'Tr_velo_to_cam': (4, 4) LiDAR to reference camera transform (homogeneous)
    """
    values = {}
    with open(calib_file) as f:
        for line in f:
            if ':' not in line:
                continue
            key, data = line.split(':', 1)
            values[key.strip()] = np.array([float(v) for v in data.split()], dtype=np.float64)

    R0_rect = np.eye(4)
    R0_rect[:3, :3] = values['R0_rect'].reshape(3, 3)

    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[:3, :4] = values['Tr_velo_to_cam'].reshape(3, 4)

    return {
        'P2': values['P2'].reshape(3, 4),
        'R0_rect': R0_rect,
        'Tr_velo_to_cam': Tr_velo_to_cam,
    }


def rect_to_velo(points_rect: np.ndarray, calib: dict) -> np.ndarray:
    """
    Transform points from the rectified camera frame to the LiDAR frame.

    Args:
        points_rect: (N, 3) points in the rectified camera frame
        calib: Calibration dict from `read_calib`

    Returns:
        (N, 3) points in the LiDAR frame
    """
    velo_to_rect = calib['R0_rect'] @ calib['Tr_velo_to_cam']
    rect_to_velo_mat = np.linalg.inv(velo_to_rect)

    points = np.hstack([points_rect, np.ones((len(points_rect), 1))])
    return (points @ rect_to_velo_mat.T)[:, :3]


def velo_to_image(points_velo: np.ndarray, calib: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Project LiDAR points into the left color image.

    Args:
        points_velo: (N, 3) points in the LiDAR frame
        calib: Calibration dict from `read_calib`

    Returns:
        uv: (N, 2) pixel coordinates (u = column, v = row)
        depth: (N,) depth in the rectified camera frame (positive in front of the camera)
    """
    points = np.hstack([points_velo, np.ones((len(points_velo), 1))])
    points_rect = points @ (calib['R0_rect'] @ calib['Tr_velo_to_cam']).T     # (N, 4)
    projected = points_rect @ calib['P2'].T                                    # (N, 3)

    depth = projected[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = projected[:, :2] / depth[:, None]
    return uv, points_rect[:, 2]


def read_label(label_file: str, calib: dict, classes: list[str]) -> np.ndarray:
    """
    Read a KITTI label file and convert the boxes of the requested classes to the LiDAR frame.

    KITTI stores [h, w, l, x, y, z, ry] with (x, y, z) the bottom center in the
    rectified camera frame and ry the rotation around the camera y axis.

    Args:
        label_file: Path to the label file
        calib: Calibration dict from `read_calib`
        classes: Object classes to keep (e.g. ['Car', 'Van'])

    Returns:
        (N, 7) float32 boxes [x, y, z_bottom, h, w, l, yaw] in the LiDAR frame
    """
    rows = []
    with open(label_file) as f:
        for line in f:
            fields = line.split()
            if not fields or fields[0] not in classes:
                continue
            rows.append([float(v) for v in fields[8:15]])

    if not rows:
        return np.zeros((0, 7), dtype=np.float32)

    rows = np.array(rows, dtype=np.float64)
    h, w, l = rows[:, 0], rows[:, 1], rows[:, 2]
    location_velo = rect_to_velo(rows[:, 3:6], calib)
    yaw = limit_period(-rows[:, 6] - np.pi / 2, offset=0.5, period=2 * np.pi)

    boxes = np.column_stack([location_velo, h, w, l, yaw])
    return boxes.astype(np.float32)


def velo_to_rect(points_velo: np.ndarray, calib: dict) -> np.ndarray:
    """
    Transform points from the LiDAR frame to the rectified camera frame.

    Args:
        points_velo: (N, 3) points in the LiDAR frame
        calib: Calibration dict from `read_calib`

    Returns:
        (N, 3) points in the rectified camera frame
    """
    points = np.hstack([points_velo, np.ones((len(points_velo), 1))])
    return (points @ (calib['R0_rect'] @ calib['Tr_velo_to_cam']).T)[:, :3]


def read_image_shape(image_file: str) -> tuple[int, int]:
    """
    Read (height, width) from a PNG header without decoding the image.
    """
    with open(image_file, 'rb') as f:
        header = f.read(24)
    if header[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError(f"Not a PNG file: {image_file}")
    width = int.from_bytes(header[16:20], 'big')
    height = int.from_bytes(header[20:24], 'big')
    return height, width


def read_label_annotations(label_file: str) -> dict:
    """
    Read every object of a KITTI label (or detection) file, as needed by the evaluation.

    Returns:
        Dict of arrays (N objects):
            'name' (N,) str, 'truncated' (N,), 'occluded' (N,), 'alpha' (N,),
            'bbox' (N, 4) [x1, y1, x2, y2], 'dimensions' (N, 3) [h, w, l],
            'location' (N, 3) bottom center in the rectified camera frame, 'rotation_y' (N,),
            and 'score' (N,) when the file has a 16th (score) column
    """
    names, values, scores = [], [], []
    with open(label_file) as f:
        for line in f:
            fields = line.split()
            if not fields:
                continue
            names.append(fields[0])
            values.append([float(v) for v in fields[1:15]])
            if len(fields) > 15:
                scores.append(float(fields[15]))

    values = np.array(values, dtype=np.float64).reshape(-1, 14)
    annos = {
        'name': np.array(names, dtype=str),
        'truncated': values[:, 0],
        'occluded': values[:, 1].astype(np.int64),
        'alpha': values[:, 2],
        'bbox': values[:, 3:7],
        'dimensions': values[:, 7:10],
        'location': values[:, 10:13],
        'rotation_y': values[:, 13],
    }
    if len(scores) == len(names):
        annos['score'] = np.array(scores, dtype=np.float64)
    return annos


def lidar_boxes_to_annotations(
    boxes: np.ndarray,
    scores: np.ndarray,
    calib: dict,
    image_shape: tuple[int, int],
    class_name: str = 'Car',
) -> dict:
    """
    Convert LiDAR-frame detections to KITTI annotations (camera frame + 2D image box).

    The 2D box is the image-clipped projection of the 8 box corners. Detections whose
    projection falls completely outside the image, or that are behind the camera,
    are dropped.

    Args:
        boxes: (N, 7) boxes [x, y, z_bottom, h, w, l, yaw] in the LiDAR frame
        scores: (N,) detection scores
        calib: Calibration dict from `read_calib`
        image_shape: (height, width) of image_2
        class_name: Class name written for every detection

    Returns:
        Annotation dict (same keys as `read_label_annotations`, plus 'score')
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    height, width = image_shape

    keep = np.zeros(len(boxes), dtype=bool)
    bbox = np.zeros((len(boxes), 4))
    if len(boxes) > 0:
        corners = boxes_to_corners_3d(boxes)                                   # (N, 8, 3)
        uv, depth = velo_to_image(corners.reshape(-1, 3), calib)
        uv, depth = uv.reshape(-1, 8, 2), depth.reshape(-1, 8)

        in_front = (depth > 0).all(axis=1)
        uv = np.where(in_front[:, None, None], uv, 0.0)
        bbox = np.concatenate([uv.min(axis=1), uv.max(axis=1)], axis=1)
        bbox[:, [0, 2]] = np.clip(bbox[:, [0, 2]], 0, width - 1)
        bbox[:, [1, 3]] = np.clip(bbox[:, [1, 3]], 0, height - 1)
        keep = in_front & (bbox[:, 2] > bbox[:, 0]) & (bbox[:, 3] > bbox[:, 1])

    boxes, scores, bbox = boxes[keep], scores[keep], bbox[keep]
    location = velo_to_rect(boxes[:, :3], calib)
    rotation_y = limit_period(-boxes[:, 6] - np.pi / 2, offset=0.5, period=2 * np.pi)
    alpha = limit_period(rotation_y - np.arctan2(location[:, 0], location[:, 2]), offset=0.5, period=2 * np.pi)

    return {
        'name': np.full(len(boxes), class_name, dtype=object).astype(str),
        'truncated': np.zeros(len(boxes)),
        'occluded': np.zeros(len(boxes), dtype=np.int64),
        'alpha': alpha,
        'bbox': bbox,
        'dimensions': boxes[:, 3:6],
        'location': location,
        'rotation_y': rotation_y,
        'score': scores,
    }


def write_annotations(annos: dict, output_file: str) -> None:
    """Write detections in the KITTI label format (with score) expected by the official devkit."""
    with open(output_file, 'w') as f:
        for i in range(len(annos['name'])):
            h, w, l = annos['dimensions'][i]
            x, y, z = annos['location'][i]
            x1, y1, x2, y2 = annos['bbox'][i]
            f.write(
                f"{annos['name'][i]} -1 -1 {annos['alpha'][i]:.4f} "
                f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {h:.4f} {w:.4f} {l:.4f} "
                f"{x:.4f} {y:.4f} {z:.4f} {annos['rotation_y'][i]:.4f} {annos['score'][i]:.4f}\n"
            )
