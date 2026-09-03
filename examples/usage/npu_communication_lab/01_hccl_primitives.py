#!/usr/bin/env python3
"""Exercise the HCCL primitives that SGLang's higher-level communication uses.

Examples:
  torchrun --standalone --nproc-per-node=16 01_hccl_primitives.py --op all
  torchrun --standalone --nproc-per-node=4 01_hccl_primitives.py --op p2p
"""

from __future__ import annotations

import argparse

import torch
import torch.distributed as dist

from lab_utils import close_dist, init_hccl, synchronized_seconds


OPS = ("broadcast", "all_reduce", "all_gather", "reduce_scatter", "all_to_all", "p2p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", choices=("all", *OPS), default="all")
    parser.add_argument("--numel", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=10)
    return parser.parse_args()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    args = parse_args()
    rank, world, local_rank = init_hccl()
    device = torch.device("npu", local_rank)
    selected = OPS if args.op == "all" else (args.op,)

    for op_name in selected:
        if op_name == "broadcast":
            tensor = torch.full((args.numel,), float(rank), device=device)

            def op():
                if rank == 0:
                    tensor.fill_(7)
                else:
                    tensor.zero_()
                dist.broadcast(tensor, src=0)

            elapsed = synchronized_seconds(op, args.iterations)
            check(bool(torch.all(tensor == 7).item()), "broadcast result mismatch")

        elif op_name == "all_reduce":
            base = torch.full((args.numel,), float(rank + 1), device=device)
            tensor = torch.empty_like(base)

            def op():
                tensor.copy_(base)
                dist.all_reduce(tensor)

            elapsed = synchronized_seconds(op, args.iterations)
            expected = world * (world + 1) / 2
            check(
                bool(torch.all(tensor == expected).item()),
                "all_reduce result mismatch",
            )

        elif op_name == "all_gather":
            local = torch.full((args.numel,), float(rank), device=device)
            gathered = torch.empty(world * args.numel, device=device)

            def op():
                dist.all_gather_into_tensor(gathered, local)

            elapsed = synchronized_seconds(op, args.iterations)
            view = gathered.view(world, args.numel)
            expected = torch.arange(world, device=device).view(world, 1).expand_as(view)
            check(bool(torch.equal(view, expected)), "all_gather result mismatch")

        elif op_name == "reduce_scatter":
            local_input = torch.full(
                (world * args.numel,), float(rank + 1), device=device
            )
            output = torch.empty(args.numel, device=device)

            def op():
                dist.reduce_scatter_tensor(output, local_input)

            elapsed = synchronized_seconds(op, args.iterations)
            expected = world * (world + 1) / 2
            check(bool(torch.all(output == expected).item()), "reduce_scatter mismatch")

        elif op_name == "all_to_all":
            # Block d on source rank s carries the value 1000*s+d to destination d.
            blocks = [
                torch.full((args.numel,), float(1000 * rank + dst), device=device)
                for dst in range(world)
            ]
            local_input = torch.cat(blocks)
            output = torch.empty_like(local_input)

            def op():
                dist.all_to_all_single(output, local_input)

            elapsed = synchronized_seconds(op, args.iterations)
            view = output.view(world, args.numel)
            expected = torch.tensor(
                [1000 * src + rank for src in range(world)],
                dtype=output.dtype,
                device=device,
            ).view(world, 1)
            check(
                bool(torch.equal(view, expected.expand_as(view))),
                "all_to_all mismatch",
            )

        else:  # p2p ring
            send = torch.full((args.numel,), float(rank), device=device)
            recv = torch.empty_like(send)
            prev_rank = (rank - 1) % world
            next_rank = (rank + 1) % world

            def op():
                requests = dist.batch_isend_irecv(
                    [
                        dist.P2POp(dist.isend, send, next_rank),
                        dist.P2POp(dist.irecv, recv, prev_rank),
                    ]
                )
                for request in requests:
                    request.wait()

            elapsed = synchronized_seconds(op, args.iterations)
            check(bool(torch.all(recv == prev_rank).item()), "P2P ring result mismatch")

        if rank == 0:
            input_elements = (
                world * args.numel
                if op_name in ("reduce_scatter", "all_to_all")
                else args.numel
            )
            print(
                f"PASS op={op_name:14s} world={world} "
                f"input_bytes_per_rank={input_elements * 4}B "
                f"max_avg_latency={elapsed * 1e3:.3f}ms",
                flush=True,
            )

    close_dist()


if __name__ == "__main__":
    main()
