#!/usr/bin/env python3
"""Simulate the dispatch -> expert compute -> combine shape of DeepEP.

This is deliberately implemented with torch.distributed all_to_all_single. DeepEP
implements the same semantic data movement with specialized routing metadata,
buffers, quantization support, and overlap-oriented kernels.
"""

from __future__ import annotations

import argparse

import torch
import torch.distributed as dist

from lab_utils import close_dist, init_hccl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-per-rank", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    args = parser.parse_args()
    rank, world, local_rank = init_hccl()
    device = torch.device("npu", local_rank)

    token_ids = rank * args.tokens_per_rank + torch.arange(
        args.tokens_per_rank, device=device
    )
    hidden = token_ids.float().unsqueeze(1).repeat(1, args.hidden)
    # A deterministic top-1 router: token i is owned by expert rank i % world.
    destinations = token_ids % world
    permutation = torch.argsort(destinations)
    packed = hidden[permutation].contiguous()
    send_counts = torch.bincount(destinations, minlength=world).to(torch.int32)
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts)
    send_splits = send_counts.cpu().tolist()
    recv_splits = recv_counts.cpu().tolist()

    dispatched = torch.empty(sum(recv_splits), args.hidden, device=device)
    dist.all_to_all_single(
        dispatched,
        packed,
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits,
    )

    # Stand-in for the local experts. The owner rank leaves an observable mark.
    expert_output = dispatched + float(rank * 1000)

    combined_packed = torch.empty_like(packed)
    dist.all_to_all_single(
        combined_packed,
        expert_output,
        output_split_sizes=send_splits,
        input_split_sizes=recv_splits,
    )
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(args.tokens_per_rank, device=device)
    combined = combined_packed[inverse]

    expected = hidden + destinations.float().unsqueeze(1) * 1000
    max_error = float((combined - expected).abs().max().cpu())
    if max_error != 0:
        raise AssertionError(f"dispatch/combine mismatch: {max_error=}")
    print(
        f"rank={rank:02d} send_counts={send_splits} recv_counts={recv_splits} PASS",
        flush=True,
    )
    close_dist()


if __name__ == "__main__":
    main()
