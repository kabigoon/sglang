#!/usr/bin/env python3
"""bench_sli.py —— 脱离模型，单独测 `SparseLightningIndexer`（A3/arch22）。

目的：把"算子本身慢"和"调用方式/数据使它慢"分开。三个实验：

  E1  S 扫描（SLI vs LIV2 同一份数据）：
        如果拐点（S≈20–24K 处 ×16）在脱离模型时**照样复现** ⇒ 算子固有；
        如果不复现 ⇒ 问题在我们的调用/数据（例如 candidate 内容或 block_table 布局）。
  E2  候选块内容（recent / sparse / none）@ 固定大 S：
        如果耗时随候选内容大幅变化 ⇒ 慢在候选掩码那条路径上（可针对性地改需求）。
  E3  T 扫描 @ 固定 S：
        验证 t ∝ T×S（残 chunk 会偏离直线）。

用法（机器上，仓库根目录）：
  # 先把张量和签名打出来，不真正调算子（最便宜的冒烟）
  python3 _tmp_preflight/bench_sli.py --dry-run

  # E1：S 扫描，SLI + LIV2 对照
  python3 _tmp_preflight/bench_sli.py --S 4096,8192,16384,20480,21504,22528,23552,24576,32768,65536,122880

  # E2：2×2 —— 小/大 S × 满/空候选（这一组最能定位"触发的到底是 S 还是掩码非平凡"）
  python3 _tmp_preflight/bench_sli.py --S 16384,20480,24576,65536 --cand-mode recent,none

  # E3：固定 S，换 T
  python3 _tmp_preflight/bench_sli.py --S 122880 --T 1024,2048,3200,4096

注意：
  * 这个脚本**只读** sglang，不改任何 python/sglang/**；它按 `_dsv41_index_qsli_topk`（BACKEND:1755）的
    实参形状构造张量（q[T,32,128] / k[pages,64,1,128] / w[T,32] / cand[T,1,2048] / block_table[1,S/64]）；
  * `topk=512`、`candidate_block_size=8`、`layout_q=TND`、`layout_k=PA_BBND`、`mask_mode=3`、`cmp_ratio=1`
    都是 trace 里实测的值；
  * 一个 chunk 的 T 是 4096；S（被扫的压缩历史长度）= block_table 宽 × 64。
"""
from __future__ import annotations

import argparse
import inspect
import statistics
import sys

import torch

IDX_HEADS = 32        # index_n_heads
IDX_DIM = 128         # index_head_dim
PAGE = 64             # k.shape[1]，压缩池页行数
TOPK = 512            # index_topk
CAND_BLOCKS = 2048    # candidate_topk_blocks
CAND_BLOCK_SIZE = 8   # candidate_block_size


def _resolve_ops():
    """优先用 sglang 走的那条 wrapper；退回到 torch.ops。"""
    ops = {}
    try:
        from sgl_kernel_npu.operators.indexer import sparse_lightning_indexer as sli

        ops["sli"] = sli
    except Exception as e:  # pragma: no cover
        print("[warn] 无法 import sgl_kernel_npu...sparse_lightning_indexer: %r" % (e,), file=sys.stderr)
    try:
        from sgl_kernel_npu.operators.indexer import lightning_indexer_v2 as liv2

        ops["liv2"] = liv2
    except Exception as e:  # pragma: no cover
        print("[warn] 无法 import ...lightning_indexer_v2: %r" % (e,), file=sys.stderr)
    for k, name in (("sli", "sparse_lightning_indexer"), ("liv2", "lightning_indexer_v2")):
        if k not in ops:
            try:
                ops[k] = getattr(torch.ops.cann_ops_transformer, name)
            except Exception:
                pass
    return ops


def sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()
    else:  # pragma: no cover
        torch.cuda.synchronize()


def build(device: str, T: int, S: int, max_pages: int, cand_mode: str = "recent"):
    """按 sglang 的实参形状造一批张量。S 必须能被 PAGE 整除（block_table 宽 = S/PAGE）。"""
    assert S % PAGE == 0, "S 必须是 %d 的倍数，got %d" % (PAGE, S)
    bt_w = S // PAGE
    assert bt_w <= max_pages, "S 太大：需要 %d 页，池只有 %d 页" % (bt_w, max_pages)

    q = torch.randn(T, IDX_HEADS, IDX_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(max_pages, PAGE, 1, IDX_DIM, device=device, dtype=torch.bfloat16)
    w = torch.rand(T, IDX_HEADS, device=device, dtype=torch.float32)
    cu_q = torch.tensor([0, T], device=device, dtype=torch.int32)
    seqused_k = torch.tensor([S], device=device, dtype=torch.int32)
    block_table = torch.arange(bt_w, device=device, dtype=torch.int32).view(1, -1)

    # candidate_topk_indices: 逻辑块号（每块 = CAND_BLOCK_SIZE 个压缩 token），-1 表示空槽
    n_logical = S // CAND_BLOCK_SIZE
    cand = torch.full((T, 1, CAND_BLOCKS), -1, device=device, dtype=torch.int32)
    if cand_mode != "none" and n_logical > 0:
        keep = min(CAND_BLOCKS, n_logical)
        ids = torch.arange(n_logical - keep, n_logical, device=device, dtype=torch.int32)
        if cand_mode == "sparse":  # 只留最近 256 块，其余填 -1
            ids = ids[-256:]
        cand[:, 0, : ids.numel()] = ids
    return q, k, w, cu_q, seqused_k, block_table, cand


KW = dict(
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    cmp_residual_k=None,
    layout_q="TND",
    layout_k="PA_BBND",
    mask_mode=3,
    cmp_ratio=1,
    max_seqlen_q=-1,
)


def run_once(op, q, k, w, cu_q, seqused_k, block_table, cand):
    return op(
        q,
        k,
        w,
        TOPK,
        cand,
        cu_seqlens_q=cu_q,
        seqused_k=seqused_k,
        block_table=block_table,
        candidate_block_size=CAND_BLOCK_SIZE,
        **KW,
    )


def timeit(fn, warmup: int, iters: int):
    import time

    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--device", default="npu")
    ap.add_argument("--S", default="4096,8192,16384,20480,24576,65536,122880",
                    help="逗号分隔的被扫历史长度（必须是 64 的倍数）")
    ap.add_argument("--T", default="4096", help="逗号分隔的 query 行数")
    ap.add_argument("--cand-mode", default="recent", help="recent,sparse,none（可逗号分隔）")
    ap.add_argument("--ops", default="sli,liv2")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true", help="只造张量并打印，不调算子")
    args = ap.parse_args()

    Ss = [int(x) for x in args.S.split(",") if x.strip()]
    Ts = [int(x) for x in args.T.split(",") if x.strip()]
    modes = [x.strip() for x in args.cand_mode.split(",") if x.strip()]
    max_pages = max(Ss) // PAGE + 8

    ops = _resolve_ops()
    if not args.dry_run and "sli" not in ops:
        print("[fatal] 找不到 sparse_lightning_indexer（既没有 sgl_kernel_npu wrapper，也没有 "
              "torch.ops.cann_ops_transformer）。请把 `--dry-run` 的输出和这个报错一起反馈。",
              file=sys.stderr)
        return 2
    for name in ("sli", "liv2"):
        if name in ops:
            try:
                print("[op] %s = %r" % (name, ops[name]), file=sys.stderr)
                print("     signature: %s" % inspect.signature(ops[name]), file=sys.stderr)
            except Exception:
                pass

    print("\n%-6s %-8s %-6s %10s %10s %10s" % ("T", "S", "cand", "SLI(ms)", "LIV2(ms)", "比"))
    for T in Ts:
        for S in Ss:
            for mode in modes:
                q, k, w, cu_q, seqused_k, bt, cand = build(args.device, T, S, max_pages, mode)
                if args.dry_run:
                    print("%-6d %-8d %-6s q%s k%s w%s bt%s cand%s" %
                          (T, S, mode, tuple(q.shape), tuple(k.shape), tuple(w.shape),
                           tuple(bt.shape), tuple(cand.shape)))
                    continue
                res = {}
                for name in args.ops.split(","):
                    name = name.strip()
                    if name not in ops:
                        continue
                    try:
                        res[name] = timeit(
                            lambda op=ops[name]: run_once(op, q, k, w, cu_q, seqused_k, bt, cand),
                            args.warmup, args.iters,
                        )
                    except Exception as e:
                        res[name] = float("nan")
                        print("[fail] %s @T=%d,S=%d: %r" % (name, T, S, e), file=sys.stderr)
                sli_ms, liv2_ms = res.get("sli", float("nan")), res.get("liv2", float("nan"))
                ratio = (sli_ms / liv2_ms) if liv2_ms and liv2_ms == liv2_ms else float("nan")
                print("%-6d %-8d %-6s %10.2f %10.2f %10.2f" % (T, S, mode, sli_ms, liv2_ms, ratio))

    if args.dry_run:
        print("\n[dry-run] 形状 OK。去掉 --dry-run 就会真正调用算子。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
