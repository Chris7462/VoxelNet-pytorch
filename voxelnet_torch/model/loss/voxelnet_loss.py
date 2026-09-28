import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VoxelNetLoss(nn.Module):
    """
    VoxelNet loss (paper, eq. 2).

        L = alpha * (1 / N_pos) * sum_pos BCE(p, 1)
          + beta  * (1 / N_neg) * sum_neg BCE(p, 0)
          + reg_weight * (1 / N_pos) * sum_pos SmoothL1(delta, target)

    Classification uses BCE on logits (numerically stable, AMP safe).
    Normalization is over all anchors of the batch.
    """

    def __init__(
        self,
        alpha: float = 1.5,
        beta: float = 1.0,
        reg_weight: float = 1.0,
        smooth_l1_beta: float = 1.0,
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.reg_weight = reg_weight
        self.smooth_l1_beta = smooth_l1_beta

    def forward(
        self,
        psm: Tensor,
        rm: Tensor,
        pos_equal_one: Tensor,
        neg_equal_one: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            psm: Score logits (B, A, H, W)
            rm: Regression map (B, 7A, H, W)
            pos_equal_one: Positive anchor mask (B, H, W, A)
            neg_equal_one: Negative anchor mask (B, H, W, A)
            targets: Regression targets (B, H, W, 7A)

        Returns:
            loss: Total loss
            loss_cls: Classification loss (for logging)
            loss_reg: Regression loss (for logging)
        """
        batch_size, num_anchors, height, width = psm.shape

        logits = psm.float().permute(0, 2, 3, 1)                                      # (B, H, W, A)
        deltas = rm.float().permute(0, 2, 3, 1).reshape(batch_size, height, width, num_anchors, 7)
        targets = targets.reshape(batch_size, height, width, num_anchors, 7)

        num_pos = pos_equal_one.sum().clamp(min=1.0)
        num_neg = neg_equal_one.sum().clamp(min=1.0)

        # BCE(p, 1) = softplus(-x), BCE(p, 0) = softplus(x)
        cls_pos = (F.softplus(-logits) * pos_equal_one).sum() / num_pos
        cls_neg = (F.softplus(logits) * neg_equal_one).sum() / num_neg
        loss_cls = self.alpha * cls_pos + self.beta * cls_neg

        reg = F.smooth_l1_loss(deltas, targets, reduction='none', beta=self.smooth_l1_beta)
        loss_reg = (reg.sum(dim=-1) * pos_equal_one).sum() / num_pos

        loss = loss_cls + self.reg_weight * loss_reg
        return loss, loss_cls, loss_reg
