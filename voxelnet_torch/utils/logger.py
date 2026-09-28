import json
from pathlib import Path


class Logger:
    """
    Training logger with history tracking.

    Tracks train / validation losses at each validation point, prints summaries,
    and writes the history to `history.json` (plots are left for later).
    """

    KEYS = ('loss', 'loss_cls', 'loss_reg', 'num_pos')

    def __init__(self, log_dir: str | Path) -> None:
        """
        Args:
            log_dir: Directory to save the history file
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.history = {'iteration': [], 'lr': []}
        for split in ('train', 'val'):
            for key in self.KEYS:
                self.history[f'{split}_{key}'] = []

    def update(self, iteration: int, lr: float, train_metrics: dict, val_metrics: dict) -> None:
        """
        Update history with metrics from one validation point.

        Args:
            iteration: Current iteration number
            lr: Current learning rate
            train_metrics: Training metrics dict with keys loss, loss_cls, loss_reg, num_pos
            val_metrics: Validation metrics dict with the same keys
        """
        self.history['iteration'].append(iteration)
        self.history['lr'].append(lr)
        for key in self.KEYS:
            self.history[f'train_{key}'].append(train_metrics[key])
            self.history[f'val_{key}'].append(val_metrics[key])

    def print_iteration(self, iteration: int, max_iter: int) -> None:
        """
        Print the summary of the latest validation point.

        Args:
            iteration: Current iteration number
            max_iter: Maximum iterations
        """
        h = self.history
        print(f"\nIteration {iteration}/{max_iter} Summary:")
        print(f"  LR: {h['lr'][-1]:.6f}")
        for split in ('train', 'val'):
            print(f"  {split.capitalize():5s} - Loss: {h[f'{split}_loss'][-1]:.4f} "
                  f"(cls: {h[f'{split}_loss_cls'][-1]:.4f}, reg: {h[f'{split}_loss_reg'][-1]:.4f}), "
                  f"pos anchors/batch: {h[f'{split}_num_pos'][-1]:.1f}")

    def save(self, file_name: str = 'history.json') -> None:
        """Write the history to a JSON file in the log directory."""
        with open(self.log_dir / file_name, 'w') as f:
            json.dump(self.history, f, indent=2)

    def get_history(self) -> dict:
        """Return the history dictionary."""
        return self.history

    def set_history(self, history: dict) -> None:
        """Set the history dictionary (for resuming training)."""
        self.history = history
