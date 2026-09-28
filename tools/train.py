import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from voxelnet_torch.datasets import KITTI
from voxelnet_torch.model import VoxelNet
from voxelnet_torch.model.loss import VoxelNetLoss
from voxelnet_torch.engine import Trainer
from voxelnet_torch.utils import compute_grid_size, load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train VoxelNet for 3D object detection')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    return parser.parse_args()


def build_dataloader(config: dict, image_set: str):
    """Build dataloader for given image set."""
    training = image_set == 'train'
    dataset = KITTI(config, image_set=image_set, training=training)

    dataloader = DataLoader(
        dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=training,
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
    """Build optimizer."""
    optimizer_cfg = config['optimizer']

    optimizer = optim.SGD(
        model.parameters(),
        lr=optimizer_cfg['lr'],
        momentum=optimizer_cfg['momentum'],
        weight_decay=optimizer_cfg['weight_decay'],
    )

    return optimizer


def build_lr_scheduler(config: dict, optimizer):
    """Build step learning-rate scheduler (stepped every iteration)."""
    lr_cfg = config['lr_scheduler']

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=lr_cfg['milestones'],
        gamma=lr_cfg['gamma'],
    )

    return scheduler


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

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Set random seed for reproducibility
    seed = config['train']['seed']
    set_seed(seed)
    print(f"Random seed: {seed}")

    # Device
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using device: {device}")

    # Build dataloaders
    print("Building dataloaders...")
    train_loader = build_dataloader(config, 'train')
    val_loader = build_dataloader(config, 'val')
    print(f"  Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_loader.dataset)} samples, {len(val_loader)} batches")

    # Build model
    print("Building model...")
    model = build_model(config).to(device)
    print(f"  Grid size (nx, ny, nz): {model.grid_size}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Build optimizer and lr scheduler
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_lr_scheduler(config, optimizer)
    print(f"  Initial LR: {optimizer.param_groups[0]['lr']:.6f}, "
          f"milestones: {config['lr_scheduler']['milestones']}")

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
    print("\nStarting training...")
    print(f"  Max iterations: {config['train']['max_iter']}")
    print(f"  Checkpoint interval: {config['checkpoint']['interval']}")
    print(f"  AMP (bfloat16): {trainer.use_amp}")
    print("=" * 60)
    trainer.train()


if __name__ == '__main__':
    main()
