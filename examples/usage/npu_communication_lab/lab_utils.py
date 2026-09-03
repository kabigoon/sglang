"""Small helpers shared by the Ascend NPU communication learning scripts."""

from __future__ import annotations

import os
import time
from typing import Callable

import torch
import torch.distributed as dist


def init_hccl() -> tuple[int, int, int]:
    """Initialize one torchrun process per NPU and return rank information."""
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "torch_npu is required. Run this script inside the SGLang "
            "Ascend environment."
        ) from exc

    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        if name not in os.environ:
            raise RuntimeError(
                f"{name} is missing. Launch with torchrun, for example: "
                "torchrun --standalone --nproc-per-node=4 script.py"
            )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", init_method="env://")
    return rank, world_size, local_rank


def synchronized_seconds(fn: Callable[[], None], iterations: int) -> float:
    """Return the maximum average device time across all ranks."""
    dist.barrier()
    torch.npu.synchronize()
    begin = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.npu.synchronize()
    elapsed = (time.perf_counter() - begin) / iterations

    elapsed_tensor = torch.tensor([elapsed], dtype=torch.float64, device="npu")
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    return float(elapsed_tensor.cpu().item())


def close_dist() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
