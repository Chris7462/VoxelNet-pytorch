"""End-to-end: synthetic KITTI files → dataset → dataloader → model → loss → trainer → checkpoint."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from voxelnet_torch.datasets import KITTI
from voxelnet_torch.engine import Trainer
from voxelnet_torch.utils import build_anchors, decode_predictions, postprocess, set_seed

from conftest import REPO_ROOT, small_config

import sys
sys.path.insert(0, REPO_ROOT)
from tools.train import build_criterion, build_lr_scheduler, build_model, build_optimizer  # noqa: E402


def test_dataset_sample_and_collate(kitti_root):
    root, gt = kitti_root
    config = small_config(root)
    dataset = KITTI(config, 'train', training=False)
    assert len(dataset) == 4

    sample = dataset[0]
    np.testing.assert_allclose(sample['gt_boxes'], gt['000000'], atol=1e-4)
    assert sample['pos_equal_one'].sum() >= 3
    assert sample['voxels'].shape[1:] == (35, 7)

    empty = dataset[3]
    assert empty['gt_boxes'].shape == (0, 7) and empty['pos_equal_one'].sum() == 0

    batch = KITTI.collate([dataset[0], dataset[3]])
    assert batch['batch_size'] == 2
    assert set(batch['coords'][:, 0].tolist()) == {0, 1}


def test_targets_decode_to_labels(kitti_root):
    """Feeding the ground-truth targets through the inference decoding recovers the labels."""
    root, gt = kitti_root
    config = small_config(root)
    dataset = KITTI(config, 'train', training=False)
    sample = dataset[0]
    anchors = torch.from_numpy(build_anchors(config))

    pos = torch.from_numpy(sample['pos_equal_one'])[None]                     # (1, H, W, A)
    psm = (pos * 20 - 10).permute(0, 3, 1, 2)                                 # confident logits
    rm = torch.from_numpy(sample['targets'])[None].permute(0, 3, 1, 2)

    scores, boxes = decode_predictions(psm, rm, anchors)
    assert (scores[0] > 0.5).sum() == pos.sum()

    dets = postprocess(psm, rm, anchors, score_threshold=0.5, nms_iou_threshold=0.1)[0]
    assert len(dets['boxes']) == len(gt['000000'])
    for box in gt['000000']:
        dist = (dets['boxes'][:, :6].double() - torch.from_numpy(box[:6])).abs().max(dim=1).values
        assert dist.min() < 1e-3


def test_augmented_training_sample(kitti_root):
    root, _ = kitti_root
    config = small_config(root)
    set_seed(0)
    dataset = KITTI(config, 'train', training=True)
    for idx in range(len(dataset)):
        sample = dataset[idx]
        assert np.isfinite(sample['targets']).all()
        assert not np.any((sample['pos_equal_one'] == 1) & (sample['neg_equal_one'] == 1))


def test_trainer_runs_and_resumes(kitti_root, tmp_path):
    root, _ = kitti_root
    config = small_config(root)
    config['train']['max_iter'] = 4
    config['checkpoint']['interval'] = 2
    config['checkpoint']['save_dir'] = str(tmp_path / 'ckpt')
    config['logging']['print_interval'] = 1
    config['lr_scheduler']['milestones'] = [3]
    config['optimizer']['lr'] = 0.005

    set_seed(0)
    train_ds = KITTI(config, 'train', training=True)
    val_ds = KITTI(config, 'val', training=False)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=KITTI.collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=2, collate_fn=KITTI.collate)

    model = build_model(config)
    optimizer = build_optimizer(config, model)
    scheduler = build_lr_scheduler(config, optimizer)
    trainer = Trainer(model, train_loader, val_loader, build_criterion(config),
                      optimizer, scheduler, config, torch.device('cpu'))
    trainer.train()

    assert (tmp_path / 'ckpt' / 'latest.pth').exists()
    assert (tmp_path / 'ckpt' / 'best.pth').exists()
    assert (tmp_path / 'ckpt' / 'history.json').exists()
    history = trainer.logger.get_history()
    assert history['iteration'] == [2, 4]
    assert all(np.isfinite(history['val_loss']))
    assert optimizer.param_groups[0]['lr'] == 0.005 * 0.1

    # Resume into a fresh trainer
    model2 = build_model(config)
    optimizer2 = build_optimizer(config, model2)
    scheduler2 = build_lr_scheduler(config, optimizer2)
    trainer2 = Trainer(model2, train_loader, val_loader, build_criterion(config),
                       optimizer2, scheduler2, config, torch.device('cpu'))
    trainer2.load_checkpoint(tmp_path / 'ckpt' / 'latest.pth')
    assert trainer2.start_iter == 4
    assert optimizer2.param_groups[0]['lr'] == optimizer.param_groups[0]['lr']
    for (k, a), b in zip(model.state_dict().items(), model2.state_dict().values()):
        assert torch.equal(a, b), k
