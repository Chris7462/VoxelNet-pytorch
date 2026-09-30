"""Optimizer / learning-rate schedule builders used by tools/train.py."""

import math
import sys

import pytest
import torch

from voxelnet_torch.engine import WarmupCosineLR

from conftest import REPO_ROOT, small_config

sys.path.insert(0, REPO_ROOT)
from tools.train import build_lr_scheduler, build_model, build_optimizer  # noqa: E402


def lrs_over(scheduler, optimizer, steps):
    out = []
    for _ in range(steps):
        out.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()
    return out


def test_warmup_cosine_values():
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=0.002, momentum=0.9)
    scheduler = WarmupCosineLR(optimizer, max_iter=1000, warmup=100, warmup_factor=0.1, min_lr=1e-5)
    lrs = lrs_over(scheduler, optimizer, 1001)

    assert lrs[0] == pytest.approx(0.0002)                      # warmup_factor x lr
    assert lrs[50] == pytest.approx(0.002 * (0.1 + 0.9 * 0.5))  # linear warmup
    assert lrs[100] == pytest.approx(0.002)                     # peak right after warmup
    assert lrs[550] == pytest.approx(1e-5 + (0.002 - 1e-5) * 0.5)
    assert lrs[1000] == pytest.approx(1e-5)                     # end of schedule
    assert all(a >= b for a, b in zip(lrs[100:], lrs[101:]))    # monotone after warmup


def test_cosine_scheduler_resumes_from_state_dict():
    param = torch.nn.Parameter(torch.zeros(1))
    opt_a = torch.optim.SGD([param], lr=0.01, momentum=0.9)
    sched_a = WarmupCosineLR(opt_a, max_iter=200, warmup=20)
    lrs_over(sched_a, opt_a, 120)

    opt_b = torch.optim.SGD([param], lr=0.01, momentum=0.9)
    sched_b = WarmupCosineLR(opt_b, max_iter=200, warmup=20)
    opt_b.load_state_dict(opt_a.state_dict())
    sched_b.load_state_dict(sched_a.state_dict())

    assert lrs_over(sched_b, opt_b, 50) == pytest.approx(lrs_over(sched_a, opt_a, 50))


def test_build_from_configs():
    config = small_config()
    model = build_model(config)

    # Default config: single parameter group, no weight decay, MultiStepLR
    optimizer = build_optimizer(config, model)
    assert len(optimizer.param_groups) == 1
    assert isinstance(build_lr_scheduler(config, optimizer), torch.optim.lr_scheduler.MultiStepLR)

    # Momentum + weight decay (conv / linear weights only) + cosine
    config['optimizer'].update(lr=0.002, momentum=0.9, weight_decay=1e-4)
    config['lr_scheduler'] = {'type': 'cosine', 'warmup': 10, 'warmup_factor': 0.1, 'min_lr': 1e-5}
    config['train']['max_iter'] = 100
    optimizer = build_optimizer(config, model)
    decay, no_decay = optimizer.param_groups
    assert decay['weight_decay'] == 1e-4 and no_decay['weight_decay'] == 0.0
    assert all(p.ndim > 1 for p in decay['params']) and all(p.ndim == 1 for p in no_decay['params'])
    assert len(decay['params']) + len(no_decay['params']) == len(list(model.parameters()))
    assert optimizer.param_groups[0]['momentum'] == 0.9

    scheduler = build_lr_scheduler(config, optimizer)
    assert isinstance(scheduler, WarmupCosineLR)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.0002)
    assert all(g['lr'] == pytest.approx(0.0002) for g in optimizer.param_groups)

    with pytest.raises(ValueError):
        build_lr_scheduler({**config, 'lr_scheduler': {'type': 'poly'}}, optimizer)


def test_shipped_cosine_config():
    from voxelnet_torch.utils import load_config
    import os
    config = load_config(os.path.join(REPO_ROOT, 'configs', 'voxelnet_kitti_car_2gpu_bs16_cosine.yaml'))
    assert config['lr_scheduler']['type'] == 'cosine'
    assert config['train']['max_iter'] == 37120
    assert config['dataloader']['batch_size'] * 2 == 16
    assert math.isclose(config['optimizer']['momentum'], 0.9)


def test_trainer_with_cosine_schedule_and_resume(kitti_root, tmp_path):
    from torch.utils.data import DataLoader
    from voxelnet_torch.datasets import KITTI
    from voxelnet_torch.engine import Trainer
    from voxelnet_torch.utils import set_seed
    from tools.train import build_criterion

    root, _ = kitti_root
    config = small_config(root)
    config['train']['max_iter'] = 6
    config['optimizer'].update(lr=0.002, momentum=0.9, weight_decay=1e-4)
    config['lr_scheduler'] = {'type': 'cosine', 'warmup': 2, 'warmup_factor': 0.1, 'min_lr': 1e-5}
    config['checkpoint'] = {'save_dir': str(tmp_path / 'ckpt'), 'interval': 3}
    config['logging']['print_interval'] = 1

    set_seed(0)
    train_loader = DataLoader(KITTI(config, 'train', training=True), batch_size=2, shuffle=True,
                              collate_fn=KITTI.collate, drop_last=True)
    val_loader = DataLoader(KITTI(config, 'val'), batch_size=2, collate_fn=KITTI.collate)

    def make_trainer():
        model = build_model(config)
        optimizer = build_optimizer(config, model)
        return Trainer(model, train_loader, val_loader, build_criterion(config), optimizer,
                       build_lr_scheduler(config, optimizer), config, torch.device('cpu'))

    trainer = make_trainer()
    trainer.max_iter = 3
    trainer.train()
    lr_at_3 = trainer.optimizer.param_groups[0]['lr']

    resumed = make_trainer()
    resumed.load_checkpoint(tmp_path / 'ckpt' / 'latest.pth')
    assert resumed.start_iter == 3
    assert resumed.optimizer.param_groups[0]['lr'] == pytest.approx(lr_at_3)
    assert len(resumed.optimizer.state) > 0                       # momentum buffers restored
    resumed.train()
    assert resumed.optimizer.param_groups[0]['lr'] == pytest.approx(1e-5)
