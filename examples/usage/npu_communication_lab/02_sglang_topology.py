#!/usr/bin/env python3
"""Print the process groups implied by SGLang's model-parallel formulas.

This script does not require an NPU. It is intentionally a topology calculator:
predict the groups here, then compare them with SGLang's server startup logs.
"""

from __future__ import annotations

import argparse


def chunks(size: int, width: int) -> list[list[int]]:
    return [list(range(i, i + width)) for i in range(0, size, width)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--tp-size", type=int, default=16)
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--attn-dp-size", type=int, default=16)
    parser.add_argument("--attn-cp-size", type=int, default=1)
    parser.add_argument("--moe-dp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=16)
    parser.add_argument("--rank", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    a = parse_args()
    if not 0 <= a.rank < a.world_size:
        raise ValueError("rank must be in [0, world_size)")
    if a.world_size != a.tp_size * a.pp_size:
        raise ValueError("world_size must equal tp_size * pp_size")
    if a.tp_size % (a.attn_dp_size * a.attn_cp_size):
        raise ValueError("tp_size must be divisible by attn_dp_size * attn_cp_size")
    if a.tp_size % (a.moe_dp_size * a.ep_size):
        raise ValueError("tp_size must be divisible by moe_dp_size * ep_size")

    attn_tp = a.tp_size // a.attn_dp_size // a.attn_cp_size
    moe_tp = a.tp_size // a.moe_dp_size // a.ep_size
    tp_groups = chunks(a.world_size, a.tp_size)
    pp_groups = [
        list(range(offset, a.world_size, a.tp_size)) for offset in range(a.tp_size)
    ]
    attn_tp_groups: list[list[int]] = []
    attn_cp_groups: list[list[int]] = []
    moe_ep_groups: list[list[int]] = []

    for tp_base in range(0, a.world_size, a.tp_size):
        for dp_idx in range(a.attn_dp_size):
            for attn_tp_idx in range(attn_tp):
                start = tp_base + dp_idx * attn_tp * a.attn_cp_size + attn_tp_idx
                attn_cp_groups.append(
                    list(range(start, start + attn_tp * a.attn_cp_size, attn_tp))
                )
        for cp_dp_idx in range(a.attn_cp_size * a.attn_dp_size):
            start = tp_base + cp_dp_idx * attn_tp
            attn_tp_groups.append(list(range(start, start + attn_tp)))
        for moe_dp_idx in range(a.moe_dp_size):
            for moe_tp_idx in range(moe_tp):
                start = tp_base + moe_dp_idx * a.ep_size * moe_tp + moe_tp_idx
                moe_ep_groups.append(
                    list(range(start, start + a.ep_size * moe_tp, moe_tp))
                )

    print(
        f"world={a.world_size}, PP={a.pp_size}, TP-per-stage={a.tp_size}, "
        f"attention=(DP={a.attn_dp_size}, CP={a.attn_cp_size}, TP={attn_tp}), "
        f"MoE=(DP={a.moe_dp_size}, EP={a.ep_size}, TP={moe_tp})"
    )
    for name, groups in (
        ("TP", tp_groups),
        ("PP", pp_groups),
        ("ATTN_TP", attn_tp_groups),
        ("ATTN_CP", attn_cp_groups),
        ("MOE_EP", moe_ep_groups),
    ):
        mine = next((group for group in groups if a.rank in group), None)
        print(f"{name:8s} rank {a.rank:2d} belongs to {mine}; all groups={groups}")


if __name__ == "__main__":
    main()
