from pathlib import Path

import torch
import torch.nn as nn

from ..utils import Logger, Metrics, infinite_loader


class Trainer:
    """
    Iteration-based trainer for VoxelNet.

    Args:
        model: VoxelNet model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function (VoxelNetLoss)
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler, stepped every iteration
        config: Configuration dictionary
        device: Device to train on
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
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.device = torch.device(device)

        # Training settings
        self.max_iter = config['train']['max_iter']
        self.checkpoint_interval = config['checkpoint']['interval']
        self.print_interval = config['logging']['print_interval']

        # Mixed precision (CUDA only): bfloat16 where the GPU supports it natively (Ampere+),
        # otherwise float16 with a GradScaler (e.g. Turing / RTX 20xx)
        self.use_amp = bool(config['train'].get('amp', False)) and self.device.type == 'cuda'
        self.amp_dtype = self._resolve_amp_dtype(config['train'].get('amp_dtype', 'auto'))
        self.scaler = torch.amp.GradScaler(
            'cuda', enabled=self.use_amp and self.amp_dtype == torch.float16
        )

        # Checkpoint settings
        self.save_dir = Path(config['checkpoint']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Logger and metrics
        self.logger = Logger(self.save_dir)
        self.metrics = Metrics()

        # Training state
        self.start_iter = 0
        self.best_val_loss = float('inf')

    def _resolve_amp_dtype(self, amp_dtype: str) -> torch.dtype:
        """Pick the autocast dtype: 'bfloat16', 'float16', or 'auto'."""
        if not self.use_amp:
            return torch.float32
        if amp_dtype == 'auto':
            native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
            return torch.bfloat16 if native_bf16 else torch.float16
        if amp_dtype not in ('bfloat16', 'float16'):
            raise ValueError(f"amp_dtype must be 'auto', 'bfloat16' or 'float16', got {amp_dtype!r}")
        return getattr(torch, amp_dtype)

    def _to_device(self, sample: dict) -> dict:
        """Move the tensors of a collated batch to the training device."""
        keys = ('voxels', 'num_points', 'coords', 'pos_equal_one', 'neg_equal_one', 'targets')
        return {key: sample[key].to(self.device, non_blocking=True) for key in keys}

    def _step(self, sample: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Forward pass and loss for one batch."""
        batch = self._to_device(sample)

        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            psm, rm = self.model(batch['voxels'], batch['num_points'], batch['coords'], sample['batch_size'])

        loss, loss_cls, loss_reg = self.criterion(
            psm, rm, batch['pos_equal_one'], batch['neg_equal_one'], batch['targets']
        )
        num_pos = batch['pos_equal_one'].sum().item()
        return loss, loss_cls, loss_reg, num_pos

    def train(self) -> None:
        """Main training loop over all iterations."""
        train_iter = iter(infinite_loader(self.train_loader))

        self.model.train()

        for cur_iter in range(self.start_iter, self.max_iter):
            sample = next(train_iter)

            # Forward pass
            self.optimizer.zero_grad(set_to_none=True)
            loss, loss_cls, loss_reg, num_pos = self._step(sample)

            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.lr_scheduler.step()

            # Update metrics
            self.metrics.update(loss.item(), loss_cls.item(), loss_reg.item(), num_pos)

            # Print training metrics
            if (cur_iter + 1) % self.print_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"Iter [{cur_iter + 1}/{self.max_iter}] LR: {lr:.6f} | "
                    f"Loss: {loss.item():.4f} (cls: {loss_cls.item():.4f}, reg: {loss_reg.item():.4f}) | "
                    f"pos anchors: {num_pos:.0f}"
                )

            # Validation and checkpoint at intervals
            if (cur_iter + 1) % self.checkpoint_interval == 0 or cur_iter + 1 == self.max_iter:
                train_metrics = self.metrics.get_metrics()
                self.metrics.reset()

                val_metrics = self.validate()

                lr = self.optimizer.param_groups[0]['lr']
                self.logger.update(cur_iter + 1, lr, train_metrics, val_metrics)
                self.logger.print_iteration(cur_iter + 1, self.max_iter)
                self.logger.save()

                # Save best first so that `latest.pth` records the updated best loss
                val_loss = val_metrics['loss']
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                self.save_checkpoint(cur_iter + 1, is_best=is_best)
                if is_best:
                    print(f"Best model saved! (loss: {val_loss:.4f})")

                print("-" * 60)

                # Switch back to training mode
                self.model.train()

        print("\nTraining completed!")

    @torch.no_grad()
    def validate(self) -> dict:
        """Compute the losses on the validation set."""
        self.model.eval()

        val_metrics = Metrics()
        print(f"Validating... ({len(self.val_loader)} batches)")

        for sample in self.val_loader:
            loss, loss_cls, loss_reg, num_pos = self._step(sample)
            val_metrics.update(loss.item(), loss_cls.item(), loss_reg.item(), num_pos)

        return val_metrics.get_metrics()

    def save_checkpoint(self, iteration: int, is_best: bool = False) -> None:
        """
        Save model checkpoint.

        Args:
            iteration: Current iteration number
            is_best: Whether this is the best model so far
        """
        state = {
            'iteration': iteration,
            'net': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.logger.get_history(),
        }

        torch.save(state, self.save_dir / 'latest.pth')

        if is_best:
            torch.save(state, self.save_dir / 'best.pth')

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """
        Load checkpoint to resume training.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])

        self.start_iter = checkpoint['iteration']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        if 'history' in checkpoint:
            self.logger.set_history(checkpoint['history'])

        print(f"  Resumed from iteration {checkpoint['iteration']}")
