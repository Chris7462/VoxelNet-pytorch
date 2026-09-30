"""
Cosine learning rate scheduler with linear warmup.
"""

import math

from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineLR(LRScheduler):
    """
    Linear warmup followed by cosine decay, stepped every iteration.

    During warmup (iteration < warmup):
        lr = base_lr * (warmup_factor + (1 - warmup_factor) * iteration / warmup)

    After warmup:
        progress = (iteration - warmup) / (max_iter - warmup)
        lr = min_lr + (base_lr - min_lr) * (1 + cos(pi * progress)) / 2

    Args:
        optimizer: Wrapped optimizer
        max_iter: Total number of training iterations
        warmup: Number of warmup iterations (default: 0)
        warmup_factor: Learning rate multiplier at iteration 0 (default: 0.1)
        min_lr: Learning rate at the end of training (default: 0)
        last_epoch: The index of the last iteration (default: -1)

    Example:
        >>> optimizer = SGD(model.parameters(), lr=0.002, momentum=0.9)
        >>> scheduler = WarmupCosineLR(optimizer, max_iter=37120, warmup=500)
        >>> for iteration in range(37120):
        >>>     train_step()
        >>>     optimizer.step()
        >>>     scheduler.step()
    """

    def __init__(
        self,
        optimizer,
        max_iter: int,
        warmup: int = 0,
        warmup_factor: float = 0.1,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        if max_iter <= warmup:
            raise ValueError(f"max_iter ({max_iter}) must be larger than warmup ({warmup})")
        self.max_iter = max_iter
        self.warmup = warmup
        self.warmup_factor = warmup_factor
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate for the current iteration."""
        iteration = self.last_epoch

        if iteration < self.warmup:
            alpha = iteration / self.warmup
            factor = self.warmup_factor + (1 - self.warmup_factor) * alpha
            return [base_lr * factor for base_lr in self.base_lrs]

        progress = min((iteration - self.warmup) / (self.max_iter - self.warmup), 1.0)
        cosine = (1 + math.cos(math.pi * progress)) / 2
        return [self.min_lr + (base_lr - self.min_lr) * cosine for base_lr in self.base_lrs]
