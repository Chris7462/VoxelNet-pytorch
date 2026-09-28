"""Multi-process (DDP) training on CPU with the gloo backend, launched through torchrun."""

import json
import os
import socket
import subprocess
import sys

import pytest
import torch
import yaml

from conftest import REPO_ROOT, small_config


def free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def run_torchrun(config_path, nproc=2, resume=None):
    cmd = [
        sys.executable, '-m', 'torch.distributed.run',
        f'--nproc_per_node={nproc}', '--master_addr=127.0.0.1', f'--master_port={free_port()}',
        os.path.join(REPO_ROOT, 'tools', 'train.py'), '--config', str(config_path),
    ]
    if resume:
        cmd += ['--resume', str(resume)]
    env = dict(os.environ, OMP_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600)


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed not available")
def test_ddp_train_and_resume(kitti_root, tmp_path):
    root, _ = kitti_root
    config = small_config(root)
    config['train']['max_iter'] = 4
    config['checkpoint']['interval'] = 2
    config['checkpoint']['save_dir'] = str(tmp_path / 'ckpt')
    config['logging']['print_interval'] = 1
    config['dataloader']['batch_size'] = 1
    config_path = tmp_path / 'ddp.yaml'
    config_path.write_text(yaml.safe_dump(config))

    result = run_torchrun(config_path)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]

    out = result.stdout
    assert 'x 2 processes (DDP)' in out
    assert 'effective batch size 2' in out
    assert out.count('Training completed!') == 1                  # printed by rank 0 only
    assert out.count('Iteration 4/4 Summary') == 1

    history = json.loads((tmp_path / 'ckpt' / 'history.json').read_text())
    assert history['iteration'] == [2, 4]

    # 4 train frames / (2 processes x batch 1) = 2 batches per process and epoch;
    # the val split (2 frames) is sharded as 1 frame per process
    checkpoint = torch.load(tmp_path / 'ckpt' / 'latest.pth', map_location='cpu')
    assert checkpoint['iteration'] == 4
    assert not any(k.startswith('module.') for k in checkpoint['net'])   # saved unwrapped

    # Resume for 2 more iterations
    config['train']['max_iter'] = 6
    config_path.write_text(yaml.safe_dump(config))
    result = run_torchrun(config_path, resume=tmp_path / 'ckpt' / 'latest.pth')
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert 'Resumed from iteration 4' in result.stdout
    assert 'Iter [5/6]' in result.stdout and 'Iter [1/6]' not in result.stdout

    history = json.loads((tmp_path / 'ckpt' / 'history.json').read_text())
    assert history['iteration'] == [2, 4, 6]
