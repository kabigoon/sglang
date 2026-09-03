#!/usr/bin/env python3
"""Send chunk-shaped PPProxyTensors through real SGLang PP groups.

Run four PP stages on four NPUs:
  torchrun --standalone --nproc-per-node=4 05_sglang_pp_p2p_demo.py
"""

from __future__ import annotations

import argparse
import os

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--max-inflight-sends", type=int, default=2)
    args = parser.parse_args()
    if min(
        args.chunks,
        args.chunk_tokens,
        args.hidden,
        args.max_inflight_sends,
    ) <= 0:
        raise ValueError("all numeric arguments must be positive")

    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("This example requires torch_npu") from exc

    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_pp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="hccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=world,
        backend="hccl",
    )
    pp_group = get_pp_group()
    pending_sends: list[tuple[list, PPProxyTensors]] = []

    for chunk_id in range(args.chunks):
        if rank == 0:
            proxy = PPProxyTensors(
                {
                    "hidden_states": torch.full(
                        (args.chunk_tokens, args.hidden),
                        float(chunk_id),
                        device=f"npu:{local_rank}",
                    ),
                    "residual": torch.zeros(
                        args.chunk_tokens, args.hidden, device=f"npu:{local_rank}"
                    ),
                }
            )
        else:
            received = pp_group.recv_tensor_dict(src=rank - 1)
            proxy = PPProxyTensors(received)

        # A visible stand-in for this stage's Transformer layers.
        proxy["hidden_states"] = proxy["hidden_states"] + float(rank + 1)
        print(
            f"stage={rank} chunk={chunk_id} hidden_mean="
            f"{float(proxy['hidden_states'].mean().cpu()):.1f}",
            flush=True,
        )

        if rank < world - 1:
            works = pp_group.send_tensor_dict(
                proxy.tensors, dst=rank + 1, async_send=True
            )
            pending_sends.append((works, proxy))  # keep payload alive until wait()
            if len(pending_sends) >= args.max_inflight_sends:
                old_works, _keepalive = pending_sends.pop(0)
                for work in old_works:
                    work.work.wait()
        else:
            expected = chunk_id + world * (world + 1) / 2
            actual = float(proxy["hidden_states"].mean().cpu())
            if actual != expected:
                raise AssertionError(
                    f"chunk {chunk_id}: expected mean {expected}, got {actual}"
                )

    for works, _keepalive in pending_sends:
        for work in works:
            work.work.wait()
    pp_group.barrier()
    if rank == world - 1:
        print("All chunks crossed every PP stage: PASS", flush=True)
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    main()
