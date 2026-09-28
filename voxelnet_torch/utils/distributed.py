"""
Helpers for multi-GPU training with DistributedDataParallel.

Launch with torchrun, which sets RANK / LOCAL_RANK / WORLD_SIZE:
    torchrun --nproc_per_node=3 tools/train.py --config <config>

Without torchrun every helper falls back to single-process behavior.
"""

import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    """True when running inside an initialized process group."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed() -> tuple[torch.device, int, int]:
    """
    Initialize the process group if launched by torchrun, and pick this process's device.

    Returns:
        device: The device this process should use
        rank: Global rank (0 when not distributed)
        world_size: Number of processes (1 when not distributed)
    """
    if 'WORLD_SIZE' not in os.environ or int(os.environ['WORLD_SIZE']) <= 1:
        device_type = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
        return torch.device(device_type), 0, 1

    local_rank = int(os.environ['LOCAL_RANK'])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        backend = 'nccl'
    else:
        device = torch.device('cpu')
        backend = 'gloo'

    dist.init_process_group(backend=backend, device_id=device if device.type == 'cuda' else None)
    return device, dist.get_rank(), dist.get_world_size()


def cleanup_distributed() -> None:
    """Destroy the process group if one was created."""
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def all_reduce_sum(values: list[float], device: torch.device) -> list[float]:
    """Sum a list of Python floats over all processes."""
    if not is_distributed():
        return list(values)
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()
