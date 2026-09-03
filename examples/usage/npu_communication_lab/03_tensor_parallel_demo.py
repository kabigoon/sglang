#!/usr/bin/env python3
"""Tiny tensor-parallel linear layers: column parallel AG and row parallel AR."""

from __future__ import annotations

import argparse

import torch
import torch.distributed as dist

from lab_utils import close_dist, init_hccl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--output", type=int, default=256)
    args = parser.parse_args()
    rank, world, local_rank = init_hccl()
    if args.hidden % world or args.output % world:
        raise ValueError("--hidden and --output must both be divisible by world size")

    generator = torch.Generator(device="cpu").manual_seed(2026)
    x = torch.randn(args.tokens, args.hidden, generator=generator).to(
        f"npu:{local_rank}"
    )
    weight = torch.randn(args.hidden, args.output, generator=generator).to(
        f"npu:{local_rank}"
    )

    # Column parallel: shard output columns, then all-gather the output features.
    output_shard = args.output // world
    w_col = weight[:, rank * output_shard : (rank + 1) * output_shard]
    y_col_local = x @ w_col
    gathered = [torch.empty_like(y_col_local) for _ in range(world)]
    dist.all_gather(gathered, y_col_local)
    y_col = torch.cat(gathered, dim=-1)

    # Row parallel: shard input features, compute partial sums, then all-reduce.
    hidden_shard = args.hidden // world
    x_row = x[:, rank * hidden_shard : (rank + 1) * hidden_shard]
    w_row = weight[rank * hidden_shard : (rank + 1) * hidden_shard, :]
    y_row = x_row @ w_row
    dist.all_reduce(y_row)

    reference = x @ weight
    col_error = float((y_col - reference).abs().max().cpu())
    row_error = float((y_row - reference).abs().max().cpu())
    if rank == 0:
        print(
            "Column-parallel: local matmul -> all_gather(output features)\n"
            "Row-parallel:    local partial sum -> all_reduce(sum)\n"
            f"max_error: column={col_error:.6f}, row={row_error:.6f}",
            flush=True,
        )
    if col_error > 0.1 or row_error > 0.1:
        raise AssertionError("TP result differs too much from the unsharded reference")
    close_dist()


if __name__ == "__main__":
    main()
