#!/usr/bin/env python3
"""A minimal causal Context Parallel attention using K/V all-gather."""

from __future__ import annotations

import argparse
import math

import torch
import torch.distributed as dist

from lab_utils import close_dist, init_hccl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    args = parser.parse_args()
    rank, world, local_rank = init_hccl()
    if args.sequence_length % world:
        raise ValueError("sequence length must be divisible by world size")

    device = torch.device("npu", local_rank)
    generator = torch.Generator(device="cpu").manual_seed(2026)
    q_full = torch.randn(args.sequence_length, args.hidden, generator=generator).to(
        device
    )
    k_full_ref = torch.randn(
        args.sequence_length, args.hidden, generator=generator
    ).to(device)
    v_full_ref = torch.randn(
        args.sequence_length, args.hidden, generator=generator
    ).to(device)
    local_s = args.sequence_length // world
    begin, end = rank * local_s, (rank + 1) * local_s
    q_local = q_full[begin:end].contiguous()
    k_local = k_full_ref[begin:end].contiguous()
    v_local = v_full_ref[begin:end].contiguous()

    k_full = torch.empty_like(k_full_ref)
    v_full = torch.empty_like(v_full_ref)
    dist.all_gather_into_tensor(k_full, k_local)
    dist.all_gather_into_tensor(v_full, v_local)

    scores = q_local @ k_full.T / math.sqrt(args.hidden)
    q_positions = torch.arange(begin, end, device=device).view(-1, 1)
    kv_positions = torch.arange(args.sequence_length, device=device).view(1, -1)
    scores.masked_fill_(kv_positions > q_positions, float("-inf"))
    local_output = torch.softmax(scores, dim=-1) @ v_full

    output = torch.empty(args.sequence_length, args.hidden, device=device)
    dist.all_gather_into_tensor(output, local_output)
    if rank == 0:
        ref_scores = q_full @ k_full_ref.T / math.sqrt(args.hidden)
        causal = torch.triu(
            torch.ones(
                args.sequence_length,
                args.sequence_length,
                dtype=torch.bool,
                device=device,
            ),
            diagonal=1,
        )
        ref_scores.masked_fill_(causal, float("-inf"))
        reference = torch.softmax(ref_scores, dim=-1) @ v_full_ref
        error = float((output - reference).abs().max().cpu())
        print(
            f"CP={world}: each rank computed {local_s} Q rows, all-gathered full K/V; "
            f"max_error={error:.6f}",
            flush=True,
        )
        if error > 0.1:
            raise AssertionError("CP output differs from full causal attention")
    close_dist()


if __name__ == "__main__":
    main()
