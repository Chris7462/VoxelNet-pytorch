import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from voxelnet_torch.datasets import KITTI
from voxelnet_torch.model import VoxelNet
from voxelnet_torch.model.loss import VoxelNetLoss
from voxelnet_torch.engine import Trainer, WarmupCosineLR
from voxelnet_torch.utils import (
    cleanup_distributed,
    compute_grid_size,
    init_distributed,
    is_distributed,
    load_config,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train VoxelNet for 3D object detection')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    return parser.parse_args()


def build_dataloader(config: dict, image_set: str):
    """
    Build dataloader for given image set.

    `dataloader.batch_size` is per process: with DDP on N GPUs the effective batch
    size is N * batch_size, and each process reads its own shard of the data.
    """
    training = image_set == 'train'
    dataset = KITTI(config, image_set=image_set, training=training)

    sampler = DistributedSampler(dataset, shuffle=training, drop_last=training) if is_distributed() else None

    dataloader = DataLoader(
        dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=training if sampler is None else False,
        sampler=sampler,
        num_workers=config['dataloader']['num_workers'],
        collate_fn=dataset.collate,
        pin_memory=True,
        drop_last=training,
    )

    return dataloader


def build_model(config: dict):
    """Build VoxelNet model."""
    data_cfg = config['dataset']
    model_cfg = config['model']

    model = VoxelNet(
        grid_size=compute_grid_size(data_cfg['point_cloud_range'], data_cfg['voxel_size']),
        num_anchors=len(config['anchor']['rotations']),
        vfe_channels=model_cfg['vfe_channels'],
        voxel_feature_dim=model_cfg['voxel_feature_dim'],
    )
    return model


def build_optimizer(config: dict, model):
    """
    Build SGD optimizer.

    Weight decay is applied to conv / linear weights only; BatchNorm parameters and
    biases (all 1-D parameters) are not decayed.
    """
    optimizer_cfg = config['optimizer']
    weight_decay = optimizer_cfg.get('weight_decay', 0.0)

    if weight_decay > 0:
        decay, no_decay = [], []
        for param in model.parameters():
            if param.requires_grad:
                (decay if param.ndim > 1 else no_decay).append(param)
        params = [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
        ]
    else:
        # Single group: keeps checkpoints of earlier runs resumable
        params = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.SGD(
        params,
        lr=optimizer_cfg['lr'],
        momentum=optimizer_cfg.get('momentum', 0.0),
        nesterov=optimizer_cfg.get('nesterov', False),
    )

    return optimizer


def build_lr_scheduler(config: dict, optimizer):
    """
    Build the learning-rate scheduler (stepped every iteration).

    lr_scheduler.type:
        multistep (default): lr x gamma at each milestone
        cosine: linear warmup, then cosine decay to min_lr at train.max_iter
    """
    lr_cfg = config['lr_scheduler']
    scheduler_type = lr_cfg.get('type', 'multistep')

    if scheduler_type == 'multistep':
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=lr_cfg['milestones'],
            gamma=lr_cfg['gamma'],
        )
    if scheduler_type == 'cosine':
        return WarmupCosineLR(
            optimizer,
            max_iter=config['train']['max_iter'],
            warmup=lr_cfg.get('warmup', 0),
            warmup_factor=lr_cfg.get('warmup_factor', 0.1),
            min_lr=lr_cfg.get('min_lr', 0.0),
        )
    raise ValueError(f"Unknown lr_scheduler.type: {scheduler_type!r}")


def build_criterion(config: dict):
    """Build loss function."""
    loss_cfg = config['loss']

    criterion = VoxelNetLoss(
        alpha=loss_cfg['alpha'],
        beta=loss_cfg['beta'],
        reg_weight=loss_cfg['reg_weight'],
        smooth_l1_beta=loss_cfg['smooth_l1_beta'],
    )

    return criterion


def main():
    args = parse_args()

    # Distributed setup (no-op unless launched with torchrun)
    device, rank, world_size = init_distributed()
    is_main = rank == 0
    log = print if is_main else (lambda *a, **k: None)

    # Load config
    config = load_config(args.config)
    log(f"Loaded config from {args.config}")

    # Set random seed for reproducibility (different augmentation stream per process;
    # DDP broadcasts rank 0's initial weights to all processes)
    seed = config['train']['seed']
    set_seed(seed + rank)
    log(f"Random seed: {seed}")

    log(f"Using device: {device.type}" + (f" x {world_size} processes (DDP)" if world_size > 1 else ""))

    # Build dataloaders
    log("Building dataloaders...")
    train_loader = build_dataloader(config, 'train')
    val_loader = build_dataloader(config, 'val')
    batch_size = config['dataloader']['batch_size']
    log(f"  Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches per process "
        f"(effective batch size {batch_size * world_size})")
    log(f"  Val: {len(val_loader.dataset)} samples, {len(val_loader)} batches per process")

    # Build model
    log("Building model...")
    model = build_model(config)
    if world_size > 1 and config['train'].get('sync_bn', False):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        log("  SyncBatchNorm: on")
    model = model.to(device)
    log(f"  Grid size (nx, ny, nz): {model.grid_size}")
    log(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None)

    # Build optimizer and lr scheduler
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_lr_scheduler(config, optimizer)
    lr_cfg = config['lr_scheduler']
    if lr_cfg.get('type', 'multistep') == 'cosine':
        schedule = f"cosine, warmup {lr_cfg.get('warmup', 0)} iterations, min_lr {lr_cfg.get('min_lr', 0.0)}"
    else:
        schedule = f"multistep, milestones {lr_cfg['milestones']}"
    opt_cfg = config['optimizer']
    log(f"  Optimizer: SGD lr {opt_cfg['lr']}, momentum {opt_cfg.get('momentum', 0.0)}, "
        f"weight decay {opt_cfg.get('weight_decay', 0.0)} | LR schedule: {schedule}")

    # Build criterion
    criterion = build_criterion(config).to(device)

    # Build trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    log("\nStarting training...")
    log(f"  Max iterations: {config['train']['max_iter']}")
    log(f"  Checkpoint interval: {config['checkpoint']['interval']}")
    log(f"  AMP (bfloat16): {trainer.use_amp}")
    log("=" * 60)
    trainer.train()

    cleanup_distributed()


if __name__ == '__main__':
    main()
