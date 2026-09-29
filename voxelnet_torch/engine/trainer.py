from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from ..utils import Logger, Metrics, barrier, get_world_size, infinite_loader, is_main_process


class Trainer:
    """
    Iteration-based trainer for VoxelNet.

    Works for single-process and DistributedDataParallel training. With DDP, every
    process runs the same loop on its own shard of the data; losses are averaged
    over all processes at each validation point, and only rank 0 prints, logs and
    writes checkpoints.

    Args:
        model: VoxelNet model (optionally wrapped in DistributedDataParallel)
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function (VoxelNetLoss)
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler, stepped every iteration
        config: Configuration dictionary
        device: Device of this process
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        config: dict,
        device: torch.device,
    ) -> None:
        self.model = model
        self.raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.device = torch.device(device)
        self.is_main = is_main_process()
        self.world_size = get_world_size()

        # Training settings
        self.max_iter = config['train']['max_iter']
        self.checkpoint_interval = config['checkpoint']['interval']
        self.print_interval = config['logging']['print_interval']

        # bfloat16 autocast (no GradScaler needed); CUDA only
        self.use_amp = bool(config['train'].get('amp', False)) and self.device.type == 'cuda'

        # Checkpoint settings
        self.save_dir = Path(config['checkpoint']['save_dir'])
        if self.is_main:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Logger and metrics
        self.logger = Logger(self.save_dir) if self.is_main else None
        self.metrics = Metrics()

        # Training state
        self.start_iter = 0
        self.best_val_loss = float('inf')

    def _print(self, *args, **kwargs) -> None:
        """Print on rank 0 only."""
        if self.is_main:
            print(*args, **kwargs)

    def _to_device(self, sample: dict) -> dict:
        """Move the tensors of a collated batch to the training device."""
        keys = ('voxels', 'num_points', 'coords', 'pos_equal_one', 'neg_equal_one', 'targets')
        return {key: sample[key].to(self.device, non_blocking=True) for key in keys}

    def _step(self, model: nn.Module, sample: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Forward pass and loss for one batch."""
        batch = self._to_device(sample)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            psm, rm = model(batch['voxels'], batch['num_points'], batch['coords'], sample['batch_size'])

        loss, loss_cls, loss_reg = self.criterion(
            psm, rm, batch['pos_equal_one'], batch['neg_equal_one'], batch['targets']
        )
        num_pos = batch['pos_equal_one'].sum().item()
        return loss, loss_cls, loss_reg, num_pos

    def train(self) -> None:
        """Main training loop over all iterations."""
        start_epoch = self.start_iter // max(len(self.train_loader), 1)
        train_iter = iter(infinite_loader(self.train_loader, start_epoch=start_epoch))

        self.model.train()

        for cur_iter in range(self.start_iter, self.max_iter):
            sample = next(train_iter)

            # Forward pass
            self.optimizer.zero_grad(set_to_none=True)
            loss, loss_cls, loss_reg, num_pos = self._step(self.model, sample)

            # Backward pass (DDP averages the gradients over all processes)
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            # Update metrics
            self.metrics.update(loss.item(), loss_cls.item(), loss_reg.item(), num_pos)

            # Print training metrics (of rank 0's batch)
            if (cur_iter + 1) % self.print_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                self._print(
                    f"Iter [{cur_iter + 1}/{self.max_iter}] LR: {lr:.6f} | "
                    f"Loss: {loss.item():.4f} (cls: {loss_cls.item():.4f}, reg: {loss_reg.item():.4f}) | "
                    f"pos anchors: {num_pos:.0f}"
                )

            # Validation and checkpoint at intervals
            if (cur_iter + 1) % self.checkpoint_interval == 0 or cur_iter + 1 == self.max_iter:
                self.metrics.all_reduce(self.device)
                train_metrics = self.metrics.get_metrics()
                self.metrics.reset()

                val_metrics = self.validate()

                val_loss = val_metrics['loss']
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                if self.is_main:
                    lr = self.optimizer.param_groups[0]['lr']
                    self.logger.update(cur_iter + 1, lr, train_metrics, val_metrics)
                    self.logger.print_iteration(cur_iter + 1, self.max_iter)
                    self.logger.save()

                    self.save_checkpoint(cur_iter + 1, is_best=is_best)
                    if is_best:
                        print(f"Best model saved! (loss: {val_loss:.4f})")
                    print("-" * 60)

                barrier()

                # Switch back to training mode
                self.model.train()

        self._print("\nTraining completed!")

    @torch.no_grad()
    def validate(self) -> dict:
        """Compute the losses on the validation set (averaged over all processes)."""
        self.raw_model.eval()

        val_metrics = Metrics()
        self._print(f"Validating... ({len(self.val_loader)} batches per process, "
                    f"{self.world_size} process(es))")

        # The unwrapped model avoids DDP buffer syncs; each process handles its own shard
        for sample in self.val_loader:
            loss, loss_cls, loss_reg, num_pos = self._step(self.raw_model, sample)
            val_metrics.update(loss.item(), loss_cls.item(), loss_reg.item(), num_pos)

        val_metrics.all_reduce(self.device)
        return val_metrics.get_metrics()

    def save_checkpoint(self, iteration: int, is_best: bool = False) -> None:
        """
        Save model checkpoint (call on rank 0 only).

        Args:
            iteration: Current iteration number
            is_best: Whether this is the best model so far
        """
        state = {
            'iteration': iteration,
            'net': self.raw_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.logger.get_history(),
        }

        torch.save(state, self.save_dir / 'latest.pth')

        if is_best:
            torch.save(state, self.save_dir / 'best.pth')

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """
        Load checkpoint to resume training (call on every process).

        Args:
            checkpoint_path: Path to checkpoint file
        """
        self._print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.raw_model.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        self.start_iter = checkpoint['iteration']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        if self.logger is not None and 'history' in checkpoint:
            self.logger.set_history(checkpoint['history'])

        self._print(f"  Resumed from iteration {checkpoint['iteration']}")
