#!/usr/bin/env python3
"""dump_indexer_calls.py —— 从 kernel_details.csv 里导出 indexer 逐次调用（含 S/T/phase），给 t(S) 曲线用。

为什么不用一行 csv filter：`S`（这次调用扫了多长的压缩历史）**不在任何列里**，只能从
`Input Shapes` 里推；而且同一个 trace 里 prefill 和 decode 会混在一起（T=4096 vs T=1），
混着拟合出来的斜率是错的。这个脚本把这两件事一起做掉。

用法：
  # 目录：自动挑 kernel_details.csv 最大的那个 rank
  python3 _tmp_preflight/dump_indexer_calls.py /tmp/prof_128k_cold_1124_124110

  # 直接给 csv
  python3 _tmp_preflight/dump_indexer_calls.py <...>/kernel_details.csv --out calls.csv

  # page_size 不是 64 时显式给（本模型压缩页是 64）
  python3 ... --page 64

输出列（缺失的输入列自动跳过）：
  Name, Start Time(us), Duration(us), dur_ms, T, S(derived), phase, Input Shapes,
  Block Num, Mix Block Num, aic_mac_time(us), aic_mac_ratio, cube_utilization(%), Type, Accelerator Core

推导规则（本仓库的算子 ABI）：
  T = Input Shapes 第一段（q 的第一个维度）
  S = block_table 宽度 × page_size；block_table 是形如 "1,1920" 的那一段（取最大的）
  phase = prefill (T>1) / decode (T==1)

注意：
  * 同一个 S 会有多次调用（每 chunk 4 次：layer 24/28/32/36），拟合时用中位数；
  * `LightningIndexerV2` 在 ratio-2 的层上 S = T/2，ratio-1 的层上 S = T —— 拟合斜率时**只取一种算子**；
  * `4065` 这类数字是 **池的页数**（容量），不是 S，别拿去当横坐标。
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import statistics
import sys

WANT = [
    "Name",
    "Start Time(us)",
    "Duration(us)",
    "Input Shapes",
    "Block Num",
    "Mix Block Num",
    "aic_mac_time(us)",
    "aic_mac_ratio",
    "cube_utilization(%)",
    "Type",
    "Accelerator Core",
]
BLOCK_TABLE_RE = re.compile(r"^1,(\d+)$")


def find_csv(root: str) -> str:
    if os.path.isfile(root):
        return root
    cands = glob.glob(os.path.join(root, "**", "kernel_details.csv"), recursive=True)
    if not cands:
        sys.exit("在 %s 下没找到 kernel_details.csv" % root)
    return max(cands, key=os.path.getsize)


_TRIPLE_RE = re.compile(r"^(\d+),(\d+),(\d+)$")


def parse_shapes(shapes: str, page: int):
    """返回 (T, S, 匹配到的 block_table 宽度)。

    q 通常是第 0 段（`T,heads,dim`），但不同 profiler 版本下第 0 段可能是空的或别的张量，
    所以退化为"扫第一个三段式 shape"。都失败时 T=0（调用方应标 unknown，而不是 decode）。
    """
    fields = [f.strip() for f in (shapes or "").split(";")]
    T = 0
    if fields and fields[0]:
        try:
            T = int(fields[0].split(",")[0])
        except ValueError:
            T = 0
    if T <= 0:
        for f in fields:
            m = _TRIPLE_RE.match(f)
            if m:
                T = int(m.group(1))
                break
    widths = []
    for f in fields:
        m = BLOCK_TABLE_RE.match(f)
        if m:
            widths.append(int(m.group(1)))
    width = max(widths) if widths else 0
    return T, width * page if width else 0, width


def lsq(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return b, my - b * mx  # slope (ms per row), intercept (ms)


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("path", help="kernel_details.csv，或包含 *_ascend_pt 的 trace 目录")
    ap.add_argument("--out", default="sglang_indexer_calls.csv")
    ap.add_argument("--page", type=int, default=64, help="压缩 KV 的 page_size（决定 S 的换算）")
    ap.add_argument("--phase", default="all", choices=["all", "prefill", "decode", "unknown"])
    ap.add_argument("--pattern", default="indexer", help="Name 过滤（不区分大小写）")
    args = ap.parse_args()

    src = find_csv(args.path)
    print("kernel_details.csv: %s (%.1f MB)" % (src, os.path.getsize(src) / 1024 / 1024), file=sys.stderr)

    rows = []
    warned = [False]
    with open(src, newline="", errors="replace") as f:
        rd = csv.DictReader(f)
        have = [c for c in WANT if c in (rd.fieldnames or [])]
        for r in rd:
            if args.pattern.lower() not in (r.get("Name") or "").lower():
                continue
            T, S, width = parse_shapes(r.get("Input Shapes") or "", args.page)
            phase = "prefill" if T > 1 else ("decode" if T == 1 else "unknown")
            if T <= 0 and not warned[0]:
                warned[0] = True
                print(
                    "[warn] 无法从 Input Shapes 解析出 T，该行标为 phase=unknown。样例行:\n"
                    "       %s | dur=%.1f ms | %r" % (r.get("Name"), dur / 1000.0, r.get("Input Shapes")),
                    file=sys.stderr,
                )
            if args.phase != "all" and phase != args.phase:
                continue
            try:
                dur = float(r.get("Duration(us)") or 0)
            except ValueError:
                dur = 0.0
            out = {c: r.get(c, "") for c in have}
            out["dur_ms"] = "%.3f" % (dur / 1000.0)
            out["T"] = T
            out["S(derived)"] = S
            out["block_table_w"] = width
            out["phase"] = phase
            rows.append(out)

    if not rows:
        sys.exit("没有匹配到任何行（检查 --pattern / --phase）")

    cols = [c for c in WANT if c in rows[0]] + ["dur_ms", "T", "S(derived)", "block_table_w", "phase"]
    cols = list(dict.fromkeys(cols))
    rows.sort(key=lambda r: float(r.get("Start Time(us)") or 0))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("wrote %s (%d rows)" % (args.out, len(rows)))

    # ---------------- 摘要：按 (name, phase, S) 聚合 + 线性拟合
    groups = {}
    for r in rows:
        groups.setdefault((r["Name"], r["phase"], r["S(derived)"]), []).append(float(r["dur_ms"]))
    print("\n%-26s %-8s %9s %4s %10s %10s" % ("Name", "phase", "S", "n", "median_ms", "min..max"))
    for (name, phase, S), ds in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        print(
            "%-26s %-8s %9d %4d %10.1f %8.1f..%.1f"
            % (name[:26], phase, S, len(ds), statistics.median(ds), min(ds), max(ds))
        )

    for name in sorted({r["Name"] for r in rows}):
        pts = []
        for (n, phase, S), ds in groups.items():
            if n == name and phase == "prefill" and S > 0:
                pts.append((S, statistics.median(ds)))
        pts.sort()
        fit = lsq([p[0] for p in pts], [p[1] for p in pts])
        if fit:
            slope, intercept = fit
            print(
                "\n[fit] %s (prefill, n_S=%d):  t(S) ≈ %.2f ms + %.3f µs/row × S"
                % (name, len(pts), intercept, slope * 1000.0)
            )
        elif pts:
            print("\n[fit] %s: 只有 1 个 S 点（%d），无法拟合斜率/截距" % (name, pts[0][0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
