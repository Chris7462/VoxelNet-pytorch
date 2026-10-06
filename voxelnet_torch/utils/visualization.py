"""
Visualization of 3D boxes: bird's-eye view of the point cloud and projection into the camera image.

All images are OpenCV images: uint8, (H, W, 3), BGR.
Boxes are [x, y, z_bottom, h, w, l, yaw] in the LiDAR frame.
"""

import cv2
import numpy as np

from .box_ops import boxes_to_bev_corners, boxes_to_corners_3d


# BGR
GT_COLOR = (0, 255, 0)        # Ground truth: green
PRED_COLOR = (0, 0, 255)      # Predictions: red
OTHER_COLOR = (0, 215, 255)   # Labeled objects of other classes: yellow
DONTCARE_COLOR = (255, 255, 0)  # DontCare regions (camera image only): cyan

# Corner indices (see `boxes_to_corners_3d`): 0-3 bottom, 4-7 top; 2, 3, 6, 7 are the front (+l/2) face
BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),     # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),     # top
    (0, 4), (1, 5), (2, 6), (3, 7),     # vertical
    (2, 7), (3, 6),                     # cross on the front face
)

NEAR_PLANE = 0.1              # meters in front of the camera
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _as_boxes(boxes) -> np.ndarray:
    if boxes is None:
        return np.zeros((0, 7), dtype=np.float64)
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 7)


def _put_text(image: np.ndarray, text: str, org: tuple[int, int], color: tuple, scale: float = 0.45) -> None:
    """Text with a dark outline so it stays readable on any background."""
    cv2.putText(image, text, org, FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, org, FONT, scale, color, 1, cv2.LINE_AA)


def _box_texts(scores, labels) -> list[str] | None:
    if labels is not None:
        return [str(label) for label in labels]
    if scores is not None:
        return [f'{score:.2f}' for score in scores]
    return None


def bev_pixels(xy: np.ndarray, point_cloud_range, resolution: float) -> np.ndarray:
    """
    LiDAR (x, y) to bird's-eye-view pixel (column, row).

    The image looks down on the scene with the driving direction (x) pointing up
    and the left side (y) on the left.

    Args:
        xy: (..., 2) LiDAR coordinates
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        resolution: Meters per pixel

    Returns:
        (..., 2) float pixel coordinates (u = column, v = row)
    """
    xy = np.asarray(xy, dtype=np.float64)
    u = (point_cloud_range[4] - xy[..., 1]) / resolution
    v = (point_cloud_range[3] - xy[..., 0]) / resolution
    return np.stack([u, v], axis=-1)


def draw_bev_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    point_cloud_range,
    resolution: float,
    color: tuple,
    scores: np.ndarray | None = None,
    thickness: int = 2,
    labels: list[str] | None = None,
) -> np.ndarray:
    """
    Draw box footprints (with a line from the center to the front) on a bird's-eye-view image, in place.
    `scores` or `labels` (one text per box) are written next to the boxes.
    """
    boxes = _as_boxes(boxes)
    if len(boxes) == 0:
        return image
    texts = _box_texts(scores, labels)

    corners = bev_pixels(boxes_to_bev_corners(boxes), point_cloud_range, resolution)      # (N, 4, 2)
    centers = corners.mean(axis=1)
    fronts = corners[:, 2:4].mean(axis=1)

    for i in range(len(boxes)):
        cv2.polylines(image, [np.round(corners[i]).astype(np.int32)], True, color, thickness, cv2.LINE_AA)
        cv2.line(image, tuple(np.round(centers[i]).astype(int).tolist()),
                 tuple(np.round(fronts[i]).astype(int).tolist()), color, thickness, cv2.LINE_AA)
        if texts is not None:
            u, v = corners[i, :, 0].max() + 3, corners[i, :, 1].min() + 10
            _put_text(image, texts[i], (int(u), int(v)), color)
    return image


def draw_bev(
    points: np.ndarray,
    point_cloud_range,
    gt_boxes: np.ndarray | None = None,
    pred_boxes: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    resolution: float = 0.1,
    other_boxes: np.ndarray | None = None,
    other_labels: list[str] | None = None,
) -> np.ndarray:
    """
    Bird's-eye view of a point cloud with ground-truth (green) and predicted (red) boxes,
    and optionally the labeled objects of other classes (yellow, thin).

    Points are drawn brighter the higher they are.

    Args:
        points: (P, 3+) LiDAR points
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        gt_boxes: (N, 7) ground-truth boxes
        pred_boxes: (M, 7) predicted boxes
        scores: (M,) prediction scores, written next to the predicted boxes
        resolution: Meters per pixel
        other_boxes: (K, 7) labeled objects that are not ground truth for the detector
        other_labels: K texts (class names) written next to `other_boxes`

    Returns:
        (H, W, 3) BGR image, H = x extent / resolution, W = y extent / resolution
    """
    x_min, y_min, z_min, x_max, y_max, z_max = [float(v) for v in point_cloud_range]
    height = int(round((x_max - x_min) / resolution))
    width = int(round((y_max - y_min) / resolution))
    image = np.zeros((height, width, 3), dtype=np.uint8)

    points = np.asarray(points)
    if len(points) > 0:
        uv = np.floor(bev_pixels(points[:, :2], point_cloud_range, resolution)).astype(np.int64)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        uv, z = uv[inside], points[inside, 2]
        brightness = 90 + 165 * np.clip((z - z_min) / max(z_max - z_min, 1e-6), 0, 1)
        # Highest point wins in each pixel
        gray = np.zeros((height, width), dtype=np.float64)
        np.maximum.at(gray, (uv[:, 1], uv[:, 0]), brightness)
        image[:] = gray[..., None].astype(np.uint8)

    draw_bev_boxes(image, other_boxes, point_cloud_range, resolution, OTHER_COLOR, thickness=1, labels=other_labels)
    draw_bev_boxes(image, gt_boxes, point_cloud_range, resolution, GT_COLOR)
    draw_bev_boxes(image, pred_boxes, point_cloud_range, resolution, PRED_COLOR, scores)
    return image


def project_boxes_to_image(boxes: np.ndarray, calib: dict) -> list[np.ndarray]:
    """
    Project the edges of 3D boxes into the left color image.

    Edges are clipped at a plane just in front of the camera, so boxes that are
    partly behind the camera are drawn correctly and boxes fully behind it give no segments.

    Args:
        boxes: (N, 7) boxes in the LiDAR frame
        calib: Calibration dict from `read_calib`

    Returns:
        List (length N) of (S, 2, 2) arrays: S visible segments, each [[u1, v1], [u2, v2]]
    """
    boxes = _as_boxes(boxes)
    velo_to_rect = calib['R0_rect'] @ calib['Tr_velo_to_cam']
    projection = np.asarray(calib['P2'], dtype=np.float64)

    segments = []
    for corners in boxes_to_corners_3d(boxes).astype(np.float64):
        rect = np.hstack([corners, np.ones((8, 1))]) @ velo_to_rect.T                   # (8, 4)
        box_segments = []
        for i, j in BOX_EDGES:
            a, b = rect[i], rect[j]
            if a[2] < NEAR_PLANE and b[2] < NEAR_PLANE:
                continue
            if a[2] < NEAR_PLANE:
                a = b + (a - b) * (b[2] - NEAR_PLANE) / (b[2] - a[2])
            elif b[2] < NEAR_PLANE:
                b = a + (b - a) * (a[2] - NEAR_PLANE) / (a[2] - b[2])
            uvw = np.stack([a, b]) @ projection.T                                      # (2, 3)
            box_segments.append(uvw[:, :2] / uvw[:, 2:3])
        segments.append(np.array(box_segments, dtype=np.float64).reshape(-1, 2, 2))
    return segments


def draw_boxes_on_image(
    image: np.ndarray,
    boxes: np.ndarray,
    calib: dict,
    color: tuple,
    scores: np.ndarray | None = None,
    thickness: int = 2,
    labels: list[str] | None = None,
) -> np.ndarray:
    """
    Draw 3D boxes as wireframes on the camera image, in place. The front face is marked with a cross.

    Args:
        image: (H, W, 3) BGR image_2
        boxes: (N, 7) boxes in the LiDAR frame
        calib: Calibration dict from `read_calib`
        color: BGR color
        scores: (N,) scores, written above the boxes
        thickness: Line thickness
        labels: N texts written above the boxes instead of the scores

    Returns:
        The image
    """
    height, width = image.shape[:2]
    texts = _box_texts(scores, labels)
    limit = 1e6     # keeps the integer conversion safe for points projected very far away

    for i, segments in enumerate(project_boxes_to_image(boxes, calib)):
        visible = []
        for segment in np.clip(segments, -limit, limit):
            p1 = tuple(int(v) for v in np.round(segment[0]))
            p2 = tuple(int(v) for v in np.round(segment[1]))
            inside, p1, p2 = cv2.clipLine((0, 0, width, height), p1, p2)
            if inside:
                cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)
                visible += [p1, p2]
        if texts is not None and visible:
            visible = np.array(visible)
            u, v = visible[:, 0].min(), max(visible[:, 1].min() - 4, 12)
            _put_text(image, texts[i], (int(u), int(v)), color)
    return image


def visualize_detections(
    points: np.ndarray,
    point_cloud_range,
    gt_boxes: np.ndarray | None = None,
    pred_boxes: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    image: np.ndarray | None = None,
    calib: dict | None = None,
    title: str | None = None,
    resolution: float = 0.1,
    other_boxes: np.ndarray | None = None,
    other_labels: list[str] | None = None,
    dontcare_bboxes: np.ndarray | None = None,
) -> np.ndarray:
    """
    One picture per frame: the camera image with the projected 3D boxes on top,
    the bird's-eye view of the point cloud below. Ground truth is green, predictions are red.

    Labeled objects that are not ground truth for the detector (other classes) can be added in
    yellow with their class name, and DontCare regions as cyan rectangles in the camera image,
    so that a red box without any other box around it is a detection of something unlabeled.

    Args:
        points: (P, 3+) LiDAR points
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        gt_boxes: (N, 7) ground-truth boxes
        pred_boxes: (M, 7) predicted boxes
        scores: (M,) prediction scores
        image: (H, W, 3) BGR camera image; without it (or without `calib`) only the bird's-eye view is drawn
        calib: Calibration dict from `read_calib`
        title: Text written in the top-left corner (e.g. the frame id)
        resolution: Bird's-eye-view meters per pixel
        other_boxes: (K, 7) labeled objects of other classes
        other_labels: K class names of `other_boxes`
        dontcare_bboxes: (D, 4) DontCare regions [x1, y1, x2, y2] in the camera image

    Returns:
        (H', W', 3) BGR image
    """
    bev = draw_bev(points, point_cloud_range, gt_boxes, pred_boxes, scores, resolution, other_boxes, other_labels)
    dontcare = np.zeros((0, 4)) if dontcare_bboxes is None else np.asarray(dontcare_bboxes).reshape(-1, 4)

    if image is None or calib is None:
        canvas = bev
    else:
        camera = np.ascontiguousarray(image[..., :3]).copy()
        for x1, y1, x2, y2 in np.round(dontcare).astype(int).tolist():
            cv2.rectangle(camera, (x1, y1), (x2, y2), DONTCARE_COLOR, 1, cv2.LINE_AA)
        draw_boxes_on_image(camera, other_boxes, calib, OTHER_COLOR, thickness=1, labels=other_labels)
        draw_boxes_on_image(camera, gt_boxes, calib, GT_COLOR)
        draw_boxes_on_image(camera, pred_boxes, calib, PRED_COLOR, scores)

        # Bird's-eye view centered below the camera image
        width = max(camera.shape[1], bev.shape[1])
        canvas = np.zeros((camera.shape[0] + bev.shape[0], width, 3), dtype=np.uint8)
        left = (width - camera.shape[1]) // 2
        canvas[:camera.shape[0], left:left + camera.shape[1]] = camera
        left = (width - bev.shape[1]) // 2
        canvas[camera.shape[0]:, left:left + bev.shape[1]] = bev

    num_gt, num_pred = len(_as_boxes(gt_boxes)), len(_as_boxes(pred_boxes))
    y = 20
    if title:
        _put_text(canvas, title, (8, y), (255, 255, 255), scale=0.55)
        y += 20
    _put_text(canvas, f'GT: {num_gt}', (8, y), GT_COLOR, scale=0.55)
    _put_text(canvas, f'Pred: {num_pred}', (8, y + 20), PRED_COLOR, scale=0.55)
    y += 40
    if other_boxes is not None:
        _put_text(canvas, f'Other: {len(_as_boxes(other_boxes))}', (8, y), OTHER_COLOR, scale=0.55)
        y += 20
    if dontcare_bboxes is not None and image is not None and calib is not None:
        _put_text(canvas, f'DontCare: {len(dontcare)}', (8, y), DONTCARE_COLOR, scale=0.55)
    return canvas
