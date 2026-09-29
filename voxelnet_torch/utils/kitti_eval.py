"""
KITTI 3D object detection evaluation (2D bbox / BEV / 3D AP, 11 and 40 recall points).

A NumPy port of the widely used `kitti_object_eval_python` (second.pytorch / OpenPCDet),
which reproduces the official C++ devkit: same difficulty definitions, class handling
(Van is ignored for Car, Person_sitting for Pedestrian), DontCare regions (2D only),
threshold sampling and AP computation. IoUs are computed in the rectified camera frame
like the reference, so no numba / CUDA is needed.

Annotations are dicts of arrays as returned by
`voxelnet_torch.datasets.kitti_io.read_label_annotations`
(dimensions as [h, w, l]); detections additionally carry 'score'.
"""

import numpy as np

MIN_HEIGHT = (40, 25, 25)
MAX_OCCLUSION = (0, 1, 2)
MAX_TRUNCATION = (0.15, 0.3, 0.5)
DIFFICULTIES = ('easy', 'moderate', 'hard')
METRICS = ('bbox', 'bev', '3d')
NUM_SAMPLE_POINTS = 41

# Minimum overlaps [bbox, bev, 3d] of the two standard settings
MIN_OVERLAPS = {
    'Car': ((0.7, 0.7, 0.7), (0.7, 0.5, 0.5)),
    'Pedestrian': ((0.5, 0.5, 0.5), (0.5, 0.25, 0.25)),
    'Cyclist': ((0.5, 0.5, 0.5), (0.5, 0.25, 0.25)),
}
# GT classes that neither count as positives nor as false positives
IGNORED_SIMILAR_CLASS = {'car': 'van', 'pedestrian': 'person_sitting'}


# ----------------------------------------------------------------------------
# Overlaps
# ----------------------------------------------------------------------------

def image_box_overlap(boxes: np.ndarray, query_boxes: np.ndarray, criterion: int = -1) -> np.ndarray:
    """
    Overlap of 2D boxes [x1, y1, x2, y2] (no "+1").

    criterion: -1 IoU, 0 intersection / area of `boxes`, 1 intersection / area of `query_boxes`.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    query_boxes = np.asarray(query_boxes, dtype=np.float64).reshape(-1, 4)

    iw = np.minimum(boxes[:, None, 2], query_boxes[None, :, 2]) - np.maximum(boxes[:, None, 0], query_boxes[None, :, 0])
    ih = np.minimum(boxes[:, None, 3], query_boxes[None, :, 3]) - np.maximum(boxes[:, None, 1], query_boxes[None, :, 1])
    inter = np.where((iw > 0) & (ih > 0), iw * ih, 0.0)

    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    query_area = (query_boxes[:, 2] - query_boxes[:, 0]) * (query_boxes[:, 3] - query_boxes[:, 1])

    if criterion == -1:
        denom = area[:, None] + query_area[None, :] - inter
    elif criterion == 0:
        denom = np.broadcast_to(area[:, None], inter.shape)
    elif criterion == 1:
        denom = np.broadcast_to(query_area[None, :], inter.shape)
    else:
        return inter
    return np.divide(inter, denom, out=np.zeros_like(inter), where=inter > 0)


def _bev_corners(annos: dict) -> np.ndarray:
    """
    Counter-clockwise BEV corners in the camera x-z plane, (N, 4, 2).
    Same convention as the reference `rbbox_to_corners` with boxes [x, z, l, w, ry].
    """
    x, z = annos['location'][:, 0], annos['location'][:, 2]
    w, l = annos['dimensions'][:, 1], annos['dimensions'][:, 2]
    ry = annos['rotation_y']

    u = np.stack([-l, l, l, -l], axis=1) / 2
    v = np.stack([-w, -w, w, w], axis=1) / 2
    cos, sin = np.cos(ry)[:, None], np.sin(ry)[:, None]
    return np.stack([cos * u + sin * v + x[:, None], -sin * u + cos * v + z[:, None]], axis=-1)


def _convex_intersection_area(subject: list, clipper: list) -> float:
    """Area of the intersection of two convex counter-clockwise polygons (Sutherland-Hodgman)."""
    output = subject
    for k in range(len(clipper)):
        if not output:
            return 0.0
        ax, ay = clipper[k]
        bx, by = clipper[(k + 1) % len(clipper)]
        ex, ey = bx - ax, by - ay

        points, output = output, []
        sx, sy = points[-1]
        s_in = ex * (sy - ay) - ey * (sx - ax) >= 0
        for px, py in points:
            p_in = ex * (py - ay) - ey * (px - ax) >= 0
            if p_in != s_in:
                # Intersection of segment s-p with the clipping line
                dx, dy = px - sx, py - sy
                denom = ex * dy - ey * dx
                t = (ey * (sx - ax) - ex * (sy - ay)) / denom if denom != 0 else 0.0
                output.append((sx + t * dx, sy + t * dy))
            if p_in:
                output.append((px, py))
            sx, sy, s_in = px, py, p_in

    if len(output) < 3:
        return 0.0
    area = 0.0
    for i in range(len(output)):
        x1, y1 = output[i]
        x2, y2 = output[(i + 1) % len(output)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2


def bev_intersection(dt_annos: dict, gt_annos: dict) -> np.ndarray:
    """Rotated BEV intersection areas between detections and GT, (num_dt, num_gt)."""
    num_dt, num_gt = len(dt_annos['name']), len(gt_annos['name'])
    inter = np.zeros((num_dt, num_gt))
    if num_dt == 0 or num_gt == 0:
        return inter

    dt_corners, gt_corners = _bev_corners(dt_annos), _bev_corners(gt_annos)

    # Only compute polygons for pairs whose enclosing rectangles overlap
    dt_min, dt_max = dt_corners.min(axis=1), dt_corners.max(axis=1)
    gt_min, gt_max = gt_corners.min(axis=1), gt_corners.max(axis=1)
    candidates = np.all(
        (np.minimum(dt_max[:, None], gt_max[None]) - np.maximum(dt_min[:, None], gt_min[None])) > 0, axis=2)

    dt_polys = [list(map(tuple, c)) for c in dt_corners.tolist()]
    gt_polys = [list(map(tuple, c)) for c in gt_corners.tolist()]
    for i, j in zip(*np.nonzero(candidates)):
        inter[i, j] = _convex_intersection_area(dt_polys[i], gt_polys[j])
    return inter


def bev_overlap(dt_annos: dict, gt_annos: dict, inter: np.ndarray | None = None) -> np.ndarray:
    """Rotated BEV IoU between detections and GT, (num_dt, num_gt)."""
    if inter is None:
        inter = bev_intersection(dt_annos, gt_annos)
    dt_area = dt_annos['dimensions'][:, 1] * dt_annos['dimensions'][:, 2]
    gt_area = gt_annos['dimensions'][:, 1] * gt_annos['dimensions'][:, 2]
    union = dt_area[:, None] + gt_area[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=inter > 0)


def box3d_overlap(dt_annos: dict, gt_annos: dict, inter: np.ndarray | None = None) -> np.ndarray:
    """3D IoU between detections and GT, (num_dt, num_gt). Boxes extend from y - h to y (camera y points down)."""
    if inter is None:
        inter = bev_intersection(dt_annos, gt_annos)
    dt_y, dt_h = dt_annos['location'][:, 1], dt_annos['dimensions'][:, 0]
    gt_y, gt_h = gt_annos['location'][:, 1], gt_annos['dimensions'][:, 0]

    ih = np.minimum(dt_y[:, None], gt_y[None, :]) - np.maximum((dt_y - dt_h)[:, None], (gt_y - gt_h)[None, :])
    inter3d = np.where((ih > 0) & (inter > 0), inter * np.maximum(ih, 0), 0.0)

    dt_vol = np.prod(dt_annos['dimensions'], axis=1)
    gt_vol = np.prod(gt_annos['dimensions'], axis=1)
    union = dt_vol[:, None] + gt_vol[None, :] - inter3d
    return np.divide(inter3d, union, out=np.zeros_like(inter3d), where=inter3d > 0)


def compute_overlaps(dt_annos: dict, gt_annos: dict) -> dict:
    """All three overlap matrices (num_dt, num_gt) for one frame."""
    inter = bev_intersection(dt_annos, gt_annos)
    return {
        'bbox': image_box_overlap(dt_annos['bbox'], gt_annos['bbox']),
        'bev': bev_overlap(dt_annos, gt_annos, inter),
        '3d': box3d_overlap(dt_annos, gt_annos, inter),
    }


# ----------------------------------------------------------------------------
# Matching
# ----------------------------------------------------------------------------

def clean_data(gt_annos: dict, dt_annos: dict, class_name: str, difficulty: int):
    """
    Label each GT / detection for one class and difficulty.

    ignored_gt: 0 = valid, 1 = ignored (similar class or too hard), -1 = other class
    ignored_dt: 0 = valid, 1 = ignored (2D box too small), -1 = other class

    Returns:
        num_valid_gt, ignored_gt (G,), ignored_dt (D,), dontcare boxes (C, 4)
    """
    current = class_name.lower()
    gt_names = np.char.lower(gt_annos['name'].astype(str))
    gt_height = gt_annos['bbox'][:, 3] - gt_annos['bbox'][:, 1]

    valid_class = np.where(gt_names == current, 1,
                           np.where(gt_names == IGNORED_SIMILAR_CLASS.get(current, '\0'), 0, -1))
    too_hard = ((gt_annos['occluded'] > MAX_OCCLUSION[difficulty])
                | (gt_annos['truncated'] > MAX_TRUNCATION[difficulty])
                | (gt_height <= MIN_HEIGHT[difficulty]))

    ignored_gt = np.full(len(gt_names), -1, dtype=np.int64)
    ignored_gt[(valid_class == 0) | ((valid_class == 1) & too_hard)] = 1
    ignored_gt[(valid_class == 1) & ~too_hard] = 0
    num_valid_gt = int(np.sum(ignored_gt == 0))

    dontcare = gt_annos['bbox'][gt_annos['name'] == 'DontCare'].reshape(-1, 4)

    dt_names = np.char.lower(dt_annos['name'].astype(str))
    dt_height = np.abs(dt_annos['bbox'][:, 3] - dt_annos['bbox'][:, 1])
    ignored_dt = np.where(dt_height < MIN_HEIGHT[difficulty], 1, np.where(dt_names == current, 0, -1))

    return num_valid_gt, ignored_gt, ignored_dt.astype(np.int64), dontcare


def tp_scores(overlap: np.ndarray, scores: np.ndarray, ignored_gt: np.ndarray,
              ignored_dt: np.ndarray, min_overlap: float) -> list[float]:
    """
    First pass of the evaluation (compute_fp=False in the reference): each GT takes the
    highest-scoring unassigned detection above `min_overlap`; returns the scores of the
    true positives, used to sample the score thresholds.
    """
    assigned = np.zeros(len(scores), dtype=bool)
    usable = ignored_dt != -1
    result = []
    for i in range(len(ignored_gt)):
        if ignored_gt[i] == -1:
            continue
        candidates = np.nonzero(usable & ~assigned & (overlap[:, i] > min_overlap))[0]
        if len(candidates) == 0:
            continue
        j = candidates[np.argmax(scores[candidates])]
        assigned[j] = True
        if ignored_gt[i] == 0 and ignored_dt[j] != 1:
            result.append(float(scores[j]))
    return result


def statistics_per_threshold(overlap: np.ndarray, scores: np.ndarray, ignored_gt: np.ndarray,
                             ignored_dt: np.ndarray, dontcare_overlap: np.ndarray | None,
                             min_overlap: float, thresholds: np.ndarray) -> np.ndarray:
    """
    Second pass (compute_fp=True in the reference), vectorized over score thresholds.

    Each GT takes, among the unassigned detections scoring >= threshold with overlap above
    `min_overlap`, the valid one with the largest overlap, or else the first ignored one.

    Returns:
        (T, 3) array of [tp, fp, fn] per threshold
    """
    T, D = len(thresholds), len(scores)
    stats = np.zeros((T, 3))
    if D == 0:
        stats[:, 2] = np.sum(ignored_gt == 0)
        return stats

    active = (scores[None, :] >= thresholds[:, None]) & (ignored_dt != -1)[None, :]     # (T, D)
    assigned = np.zeros((T, D), dtype=bool)
    rows = np.arange(T)

    for i in range(len(ignored_gt)):
        if ignored_gt[i] == -1:
            continue
        eligible = active & ~assigned & (overlap[:, i] > min_overlap)[None, :]

        valid = eligible & (ignored_dt == 0)[None, :]
        best_valid = np.argmax(np.where(valid, overlap[None, :, i], -np.inf), axis=1)
        has_valid = valid.any(axis=1)

        ignored_det = eligible & (ignored_dt == 1)[None, :]
        first_ignored = np.argmax(ignored_det, axis=1)
        has_ignored = ignored_det.any(axis=1)

        matched = has_valid | has_ignored
        det_idx = np.where(has_valid, best_valid, first_ignored)

        if ignored_gt[i] == 0:
            stats[~matched, 2] += 1                                     # fn
            stats[matched & has_valid, 0] += 1                          # tp
        assigned[rows[matched], det_idx[matched]] = True

    counted = active & ~assigned & (ignored_dt == 0)[None, :]
    stats[:, 1] = counted.sum(axis=1)                                   # fp

    # 2D only: unmatched detections inside DontCare regions are not false positives
    if dontcare_overlap is not None and dontcare_overlap.shape[1] > 0:
        in_dontcare = (dontcare_overlap > min_overlap).any(axis=1)
        stats[:, 1] -= (counted & in_dontcare[None, :]).sum(axis=1)

    return stats


def get_thresholds(scores: np.ndarray, num_gt: int, num_sample_pts: int = NUM_SAMPLE_POINTS) -> np.ndarray:
    """Score thresholds at (approximately) evenly spaced recall levels, as in the devkit."""
    scores = np.sort(np.asarray(scores, dtype=np.float64))[::-1]
    current_recall = 0.0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        r_recall = (i + 2) / num_gt if i < len(scores) - 1 else l_recall
        if (r_recall - current_recall) < (current_recall - l_recall) and i < len(scores) - 1:
            continue
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return np.array(thresholds)


# ----------------------------------------------------------------------------
# AP
# ----------------------------------------------------------------------------

def eval_class(gt_annos: list[dict], dt_annos: list[dict], overlaps: list[dict],
               class_name: str, metric: str, difficulty: int, min_overlap: float) -> np.ndarray:
    """
    Interpolated precision at the sampled recall levels for one class / metric / difficulty.

    Returns:
        (41,) precision array (monotonically non-increasing)
    """
    frames = []
    num_valid_gt = 0
    all_tp_scores = []
    for gt, dt, ov in zip(gt_annos, dt_annos, overlaps):
        n_valid, ignored_gt, ignored_dt, dontcare = clean_data(gt, dt, class_name, difficulty)
        num_valid_gt += n_valid
        dontcare_overlap = image_box_overlap(dt['bbox'], dontcare, criterion=0) if metric == 'bbox' else None
        frame = (ov[metric], dt['score'], ignored_gt, ignored_dt, dontcare_overlap)
        frames.append(frame)
        all_tp_scores += tp_scores(ov[metric], dt['score'], ignored_gt, ignored_dt, min_overlap)

    precision = np.zeros(NUM_SAMPLE_POINTS)
    if num_valid_gt == 0 or not all_tp_scores:
        return precision

    thresholds = get_thresholds(np.array(all_tp_scores), num_valid_gt)
    stats = np.zeros((len(thresholds), 3))
    for overlap, scores, ignored_gt, ignored_dt, dontcare_overlap in frames:
        if len(ignored_gt) == 0 and len(scores) == 0:
            continue
        stats += statistics_per_threshold(overlap, scores, ignored_gt, ignored_dt,
                                          dontcare_overlap, min_overlap, thresholds)

    tp, fp = stats[:, 0], stats[:, 1]
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    prec = np.maximum.accumulate(prec[::-1])[::-1]                      # max over higher recall
    precision[:len(prec)] = prec
    return precision


def ap_r11(precision: np.ndarray) -> float:
    """AP over 11 recall points (0, 0.1, ..., 1.0), in percent."""
    return float(precision[0::4].sum() / 11 * 100)


def ap_r40(precision: np.ndarray) -> float:
    """AP over 40 recall points (1/40, ..., 1.0; KITTI's current official metric), in percent."""
    return float(precision[1:].sum() / 40 * 100)


def evaluate(gt_annos: list[dict], dt_annos: list[dict], classes: tuple[str, ...] = ('Car',)) -> dict:
    """
    KITTI evaluation.

    Args:
        gt_annos: Ground-truth annotations, one dict per frame (all objects, incl. DontCare)
        dt_annos: Detections, one dict per frame (same order), with 'score'
        classes: Classes to evaluate

    Returns:
        Nested dict: results[class][f'{min_overlaps}'][metric] = {
            'R11': [easy, moderate, hard], 'R40': [easy, moderate, hard]}
        where min_overlaps is e.g. '0.70_0.70_0.70' ([bbox, bev, 3d] IoU thresholds)
    """
    assert len(gt_annos) == len(dt_annos), "One detection dict per GT frame is required"
    overlaps = [compute_overlaps(dt, gt) for gt, dt in zip(gt_annos, dt_annos)]

    results = {}
    for class_name in classes:
        results[class_name] = {}
        for min_overlaps in MIN_OVERLAPS[class_name]:
            key = '_'.join(f'{o:.2f}' for o in min_overlaps)
            results[class_name][key] = {}
            for metric, min_overlap in zip(METRICS, min_overlaps):
                r11, r40 = [], []
                for difficulty in range(3):
                    precision = eval_class(gt_annos, dt_annos, overlaps, class_name,
                                           metric, difficulty, min_overlap)
                    r11.append(ap_r11(precision))
                    r40.append(ap_r40(precision))
                results[class_name][key][metric] = {'R11': r11, 'R40': r40}
    return results


def format_results(results: dict) -> str:
    """Human-readable table in the style of the reference implementation."""
    lines = []
    for class_name, per_overlap in results.items():
        for key, per_metric in per_overlap.items():
            overlaps = key.replace('_', ', ')
            for recall in ('R11', 'R40'):
                lines.append(f"{class_name} AP_{recall}@{overlaps}:  (easy, moderate, hard)")
                for metric in METRICS:
                    easy, mod, hard = per_metric[metric][recall]
                    lines.append(f"  {metric:4s} AP: {easy:7.4f}, {mod:7.4f}, {hard:7.4f}")
    return '\n'.join(lines)
