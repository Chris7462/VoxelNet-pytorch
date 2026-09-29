"""KITTI AP evaluation: overlaps, matching (against a direct port of the reference), and AP values."""

import numpy as np
import pytest

from voxelnet_torch.utils import kitti_eval as ke


def annos(name, bbox, dims_hwl, loc, ry, trunc=None, occ=None, score=None):
    n = len(name)
    out = {
        'name': np.array(name, dtype=str),
        'truncated': np.zeros(n) if trunc is None else np.asarray(trunc, float),
        'occluded': np.zeros(n, dtype=np.int64) if occ is None else np.asarray(occ, np.int64),
        'alpha': np.zeros(n),
        'bbox': np.asarray(bbox, float).reshape(-1, 4),
        'dimensions': np.asarray(dims_hwl, float).reshape(-1, 3),
        'location': np.asarray(loc, float).reshape(-1, 3),
        'rotation_y': np.asarray(ry, float).reshape(-1),
    }
    if score is not None:
        out['score'] = np.asarray(score, float)
    return out


def test_bev_and_3d_iou_known_values():
    gt = annos(['Car'], [[0, 0, 100, 100]], [[1.5, 2.0, 4.0]], [[0, 1.5, 20]], [0.0])
    same = annos(['Car'], [[0, 0, 100, 100]], [[1.5, 2.0, 4.0]], [[0, 1.5, 20]], [0.0], score=[1])
    rotated = annos(['Car'], [[0, 0, 100, 100]], [[1.5, 2.0, 4.0]], [[0, 1.5, 20]], [np.pi / 2], score=[1])
    lifted = annos(['Car'], [[0, 0, 100, 100]], [[1.5, 2.0, 4.0]], [[0, 1.0, 20]], [0.0], score=[1])

    np.testing.assert_allclose(ke.bev_overlap(same, gt), [[1.0]])
    # 4 x 2 box vs. the same box rotated by 90 degrees: 2 x 2 / (8 + 8 - 4)
    np.testing.assert_allclose(ke.bev_overlap(rotated, gt), [[1 / 3]])
    # Shifted up by 0.5 m: height overlap 1.0 of 1.5
    np.testing.assert_allclose(ke.box3d_overlap(lifted, gt), [[(8 * 1.0) / (12 + 12 - 8)]])
    np.testing.assert_allclose(ke.bev_overlap(lifted, gt), [[1.0]])


def reference_compute_statistics(overlaps, dt_scores, ignored_gt, ignored_det, dontcare_overlap,
                                 min_overlap, thresh):
    """Direct Python port of `compute_statistics_jit` (compute_fp=True) from kitti_object_eval_python."""
    det_size, gt_size = len(dt_scores), len(ignored_gt)
    assigned = [False] * det_size
    ignored_threshold = [dt_scores[i] < thresh for i in range(det_size)]
    NO_DETECTION = -10000000
    tp = fp = fn = 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx, valid_detection, max_overlap, assigned_ignored_det = -1, NO_DETECTION, 0, False
        for j in range(det_size):
            if ignored_det[j] == -1 or assigned[j] or ignored_threshold[j]:
                continue
            overlap = overlaps[j, i]
            if overlap > min_overlap and (overlap > max_overlap or assigned_ignored_det) and ignored_det[j] == 0:
                max_overlap, det_idx, valid_detection, assigned_ignored_det = overlap, j, 1, False
            elif overlap > min_overlap and valid_detection == NO_DETECTION and ignored_det[j] == 1:
                det_idx, valid_detection, assigned_ignored_det = j, 1, True
        if valid_detection == NO_DETECTION and ignored_gt[i] == 0:
            fn += 1
        elif valid_detection != NO_DETECTION and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1):
            assigned[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            assigned[det_idx] = True
    for i in range(det_size):
        if not (assigned[i] or ignored_det[i] == -1 or ignored_det[i] == 1 or ignored_threshold[i]):
            fp += 1
    nstuff = 0
    if dontcare_overlap is not None:
        for i in range(dontcare_overlap.shape[1]):
            for j in range(det_size):
                if assigned[j] or ignored_det[j] in (-1, 1) or ignored_threshold[j]:
                    continue
                if dontcare_overlap[j, i] > min_overlap:
                    assigned[j] = True
                    nstuff += 1
    return tp, fp - nstuff, fn


@pytest.mark.parametrize('seed', range(20))
def test_vectorized_statistics_match_reference(seed):
    rng = np.random.default_rng(seed)
    D, G, C = rng.integers(0, 12), rng.integers(1, 8), rng.integers(0, 3)
    overlaps = rng.choice([0.0, 0.3, 0.55, 0.72, 0.8, 0.95], size=(D, G))
    scores = rng.uniform(0, 1, D)
    ignored_gt = rng.choice([-1, 0, 0, 1], size=G)
    ignored_dt = rng.choice([-1, 0, 0, 0, 1], size=D)
    dontcare = rng.choice([0.0, 0.8], size=(D, C))
    thresholds = np.sort(rng.uniform(0, 1, 7))[::-1]

    stats = ke.statistics_per_threshold(overlaps, scores, ignored_gt, ignored_dt, dontcare, 0.7, thresholds)
    for t, thresh in enumerate(thresholds):
        expected = reference_compute_statistics(overlaps, scores, ignored_gt, ignored_dt, dontcare, 0.7, thresh)
        assert tuple(stats[t].astype(int)) == expected


def test_perfect_detections_give_full_ap():
    # Like the devkit, the recall sampling assumes >= 40 valid GT objects
    rng = np.random.default_rng(0)
    gts, dts = [], []
    for _ in range(30):
        n = 4
        loc = np.column_stack([rng.uniform(-10, 10, n), np.full(n, 1.7), 10 + 12 * np.arange(n)])
        dims = np.tile([1.5, 1.6, 3.9], (n, 1))
        ry = rng.uniform(-np.pi, np.pi, n)
        bbox = np.column_stack([rng.uniform(0, 1000, n), np.full(n, 150.0), np.zeros(n), np.zeros(n)])
        bbox[:, 2] = bbox[:, 0] + 90
        bbox[:, 3] = bbox[:, 1] + 60                          # 60 px tall: easy / moderate / hard
        names = ['Car', 'Car', 'Van', 'DontCare']
        gts.append(annos(names, bbox, dims, loc, ry))
        # Detect the two cars and the van (the van must not count as a false positive)
        dts.append(annos(['Car'] * 3, bbox[:3], dims[:3], loc[:3], ry[:3], score=rng.uniform(0.5, 1, 3)))

    results = ke.evaluate(gts, dts, ('Car',))
    for key in results['Car']:
        for metric in ke.METRICS:
            np.testing.assert_allclose(results['Car'][key][metric]['R40'], 100.0)
            np.testing.assert_allclose(results['Car'][key][metric]['R11'], 100.0)


def test_no_detections_give_zero_ap():
    gt = annos(['Car'], [[0, 100, 90, 160]], [[1.5, 1.6, 3.9]], [[0, 1.7, 20]], [0.0])
    dt = annos([], np.zeros((0, 4)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), score=np.zeros(0))
    results = ke.evaluate([gt], [dt], ('Car',))
    assert results['Car']['0.70_0.70_0.70']['3d']['R40'] == [0.0, 0.0, 0.0]


def test_false_positives_lower_precision():
    # 50 frames, each with one car detected with score 0.6 and one false positive with score 0.9:
    # precision is 0.5 at every recall level
    gts, dts = [], []
    for _ in range(50):
        gts.append(annos(['Car'], [[0, 100, 90, 160]], [[1.5, 1.6, 3.9]], [[0, 1.7, 20]], [0.0]))
        dts.append(annos(['Car', 'Car'], [[0, 100, 90, 160], [500, 100, 590, 160]], [[1.5, 1.6, 3.9]] * 2,
                         [[0, 1.7, 20], [10, 1.7, 30]], [0.0, 0.0], score=[0.6, 0.9]))
    results = ke.evaluate(gts, dts, ('Car',))
    np.testing.assert_allclose(results['Car']['0.70_0.70_0.70']['bev']['R40'], 50.0)
    np.testing.assert_allclose(results['Car']['0.70_0.70_0.70']['bev']['R11'], 50.0)


def test_few_gt_caps_ap_like_devkit():
    """With fewer than 40 valid GT objects the devkit's recall sampling stops early."""
    gt = annos(['Car'], [[0, 100, 90, 160]], [[1.5, 1.6, 3.9]], [[0, 1.7, 20]], [0.0])
    dt = annos(['Car'], [[0, 100, 90, 160]], [[1.5, 1.6, 3.9]], [[0, 1.7, 20]], [0.0], score=[0.9])
    results = ke.evaluate([gt], [dt], ('Car',))
    np.testing.assert_allclose(results['Car']['0.70_0.70_0.70']['3d']['R40'], 0.0)
    np.testing.assert_allclose(results['Car']['0.70_0.70_0.70']['3d']['R11'], 100 / 11)
