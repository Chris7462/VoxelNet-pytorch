import torch
from torch import Tensor
from torchvision.ops import nms

from .box_ops import decode_boxes, boxes_to_standup_bev_torch


def decode_predictions(psm: Tensor, rm: Tensor, anchors: Tensor) -> tuple[Tensor, Tensor]:
    """
    Convert raw RPN output into scores and boxes for every anchor.

    Args:
        psm: Probability score map logits (B, A, H, W)
        rm: Regression map (B, 7A, H, W)
        anchors: Anchors (H, W, A, 7)

    Returns:
        scores: (B, H*W*A) probabilities
        boxes: (B, H*W*A, 7) decoded boxes
    """
    batch_size = psm.shape[0]

    scores = torch.sigmoid(psm.float()).permute(0, 2, 3, 1).reshape(batch_size, -1)
    deltas = rm.float().permute(0, 2, 3, 1).reshape(batch_size, -1, 7)
    boxes = decode_boxes(deltas, anchors.reshape(1, -1, 7).to(deltas))

    return scores, boxes


@torch.no_grad()
def postprocess(
    psm: Tensor,
    rm: Tensor,
    anchors: Tensor,
    score_threshold: float = 0.5,
    nms_iou_threshold: float = 0.1,
    max_detections: int = 100,
) -> list[dict]:
    """
    Decode predictions, threshold scores and run NMS on BEV standup boxes.

    Args:
        psm: Probability score map logits (B, A, H, W)
        rm: Regression map (B, 7A, H, W)
        anchors: Anchors (H, W, A, 7)
        score_threshold: Minimum score to keep a box
        nms_iou_threshold: IoU threshold for NMS
        max_detections: Maximum number of boxes kept per sample

    Returns:
        List (length B) of dicts with 'boxes' (K, 7) and 'scores' (K,)
    """
    scores, boxes = decode_predictions(psm, rm, anchors)

    results = []
    for sample_scores, sample_boxes in zip(scores, boxes):
        keep = sample_scores > score_threshold
        sample_scores, sample_boxes = sample_scores[keep], sample_boxes[keep]

        keep = nms(boxes_to_standup_bev_torch(sample_boxes), sample_scores, nms_iou_threshold)
        keep = keep[:max_detections]

        results.append({'boxes': sample_boxes[keep], 'scores': sample_scores[keep]})

    return results
