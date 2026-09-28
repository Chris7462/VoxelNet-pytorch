import numpy as np

from ..utils.box_ops import boxes_to_standup_bev, encode_boxes, iou_2d


class TargetAssigner:
    """
    Match anchors to ground-truth boxes and build the RPN training targets.

    Matching uses the IoU of the bird's-eye-view standup (axis-aligned) rectangles:
        - positive: IoU > pos_iou with some GT, or the best anchor of a GT
        - negative: IoU < neg_iou with every GT (and not positive)
        - everything else is ignored
    Each positive anchor regresses the GT it overlaps most (or the GT it is the best anchor of).

    Args:
        anchors: (H, W, A, 7) anchors
        pos_iou: Positive IoU threshold
        neg_iou: Negative IoU threshold
    """

    def __init__(self, anchors: np.ndarray, pos_iou: float, neg_iou: float) -> None:
        self.shape = anchors.shape[:3]                      # (H, W, A)
        self.anchors = anchors.reshape(-1, 7).astype(np.float64)
        self.anchors_standup = boxes_to_standup_bev(self.anchors)
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou

    def __call__(self, gt_boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            gt_boxes: (G, 7) ground-truth boxes (G may be 0)

        Returns:
            pos_equal_one: (H, W, A) float32, 1 for positive anchors
            neg_equal_one: (H, W, A) float32, 1 for negative anchors
            targets: (H, W, A * 7) float32 regression targets (0 for non-positive anchors)
        """
        H, W, A = self.shape
        num_anchors = len(self.anchors)

        pos = np.zeros(num_anchors, dtype=bool)
        neg = np.ones(num_anchors, dtype=bool)
        targets = np.zeros((num_anchors, 7), dtype=np.float32)

        if len(gt_boxes) > 0:
            gt_boxes = np.asarray(gt_boxes, dtype=np.float64)
            iou = iou_2d(self.anchors_standup, boxes_to_standup_bev(gt_boxes))    # (N, G)

            anchor_gt = iou.argmax(axis=1)
            max_iou = iou[np.arange(num_anchors), anchor_gt]
            pos = max_iou > self.pos_iou
            neg = max_iou < self.neg_iou

            # Force the best anchor of every GT to be positive
            gt_ids = np.arange(len(gt_boxes))
            best_anchor = iou.argmax(axis=0)
            has_overlap = iou[best_anchor, gt_ids] > 0
            pos[best_anchor[has_overlap]] = True
            anchor_gt[best_anchor[has_overlap]] = gt_ids[has_overlap]
            neg[pos] = False

            targets[pos] = encode_boxes(gt_boxes[anchor_gt[pos]], self.anchors[pos])

        return (
            pos.reshape(H, W, A).astype(np.float32),
            neg.reshape(H, W, A).astype(np.float32),
            targets.reshape(H, W, A * 7),
        )
