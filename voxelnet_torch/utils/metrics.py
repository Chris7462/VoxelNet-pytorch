from .distributed import all_reduce_sum


class Metrics:
    """
    Running averages of the training / validation losses.

    Tracks the total loss, its classification and regression parts,
    and the average number of positive anchors per batch.
    """

    KEYS = ('loss', 'loss_cls', 'loss_reg', 'num_pos')

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all accumulators."""
        self.sums = {key: 0.0 for key in self.KEYS}
        self.count = 0

    def update(self, loss: float, loss_cls: float, loss_reg: float, num_pos: float) -> None:
        """
        Accumulate the values of one batch.

        Args:
            loss: Total loss
            loss_cls: Classification loss
            loss_reg: Regression loss
            num_pos: Number of positive anchors in the batch
        """
        self.sums['loss'] += loss
        self.sums['loss_cls'] += loss_cls
        self.sums['loss_reg'] += loss_reg
        self.sums['num_pos'] += num_pos
        self.count += 1

    def all_reduce(self, device) -> None:
        """Sum the accumulators over all processes (no-op when not distributed)."""
        values = all_reduce_sum([self.sums[key] for key in self.KEYS] + [float(self.count)], device)
        self.sums = dict(zip(self.KEYS, values[:-1]))
        self.count = int(values[-1])

    def get_metrics(self) -> dict:
        """Return the averages since the last reset."""
        count = max(self.count, 1)
        return {key: value / count for key, value in self.sums.items()}
