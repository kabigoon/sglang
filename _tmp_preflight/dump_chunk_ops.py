#!/usr/bin/env python3
"""dump_chunk_ops.py —— 把 trace 的 kernel 序列折叠成「连续同名 run」，用于对齐计算过程/找融合点。

为什么要它：Insight 里能看形状，但要回答"这 4 个 indexer 之间那 36 层到底做了什么、
哪些小算子相邻可融合、哪里有空窗"，需要一份**顺序 + 间隔**的文本。

用法：
  python3 _tmp_preflight/dump_chunk_ops.py /tmp/demo_ep_late_1124_135951 --out /tmp/chunk_ops.txt
  python3 _tmp_preflight/dump_chunk_ops.py <kernel_details.csv> --stream 34 --max-runs 1200

输出：
  ① attention 类 kernel 的位置/时长（定位 layer 结构与 indexer 分布）
  ② 折叠后的 run 表：序号 / 相对起始(ms) / 与上一 run 的空窗(ms) / 次数 / 总耗时(ms) / 名字
  ③ 空窗统计（>1ms 的间隔累计），用于判断"双流/overlap"有多少可捡
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict

ATT = re.compile(r"attention|attn|mla|indexer", re.I)


def pick_csv(root: str) -> str:
    if os.path.isfile(root):
        return root
    cands = sorted(glob.glob(os.path.join(root, "*_ascend_pt")), key=lambda d: -os.path.getsize(
        os.path.join(d, "ASCEND_PROFILER_OUTPUT", "kernel_details.csv"))) if glob.glob(
        os.path.join(root, "*_ascend_pt")) else []
    if not cands:
        sys.exit("找不到 *_ascend_pt：%s" % root)
    print("用最有料的 rank 目录：%s" % os.path.basename(cands[0]))
    return os.path.join(cands[0], "ASCEND_PROFILER_OUTPUT", "kernel_details.csv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--stream", default=None, help="默认自动选 AI 核耗时最多的流")
    ap.add_argument("--max-runs", type=int, default=1200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = pick_csv(args.root)
    rows, per_stream = [], defaultdict(float)
    with open(path, newline="", errors="replace") as f:
        rd = csv.DictReader(f)
        for r in rd:
            try:
                s = float(r["Start Time(us)"])
                d = float(r["Duration(us)"])
            except (TypeError, ValueError, KeyError):
                continue
            sid = (r.get("Stream ID") or "?").strip()
            core = (r.get("Accelerator Core") or "").upper()
            if "CPU" not in core and "COMM" not in core and sid != "N/A":
                per_stream[sid] += d
            rows.append((sid, s, d, (r.get("Name") or "").strip(), core))
    if not rows:
        sys.exit("没有可用 kernel 行")
    sid = args.stream or max(per_stream.items(), key=lambda kv: kv[1])[0]
    main = sorted([r for r in rows if r[0] == sid], key=lambda r: r[1])
    t0 = main[0][1]
    print("流 %s：%d 个 kernel，跨度 %.1f ms" % (sid, len(main), (main[-1][1] - t0) / 1000))

    # ① attention 类位置
    att = [r for r in main if ATT.search(r[3])]
    print("\n=== attention 类 kernel（%d 个）：相对起始 ms / 时长 ms ===" % len(att))
    for r in att[:80]:
        print("  %10.1f  %9.3f  %s" % ((r[1] - t0) / 1000, r[2] / 1000, r[3][:60]))
    if len(att) > 80:
        print("  …（其余 %d 个略）" % (len(att) - 80))

    # ② 折叠连续同名
    runs = []
    for _sid, s, d, n, _c in main:
        if runs and runs[-1][1] == n:
            runs[-1][2] += 1
            runs[-1][3] += d
            runs[-1][4] = s + d
        else:
            runs.append([s, n, 1, d, s + d])

    # ③ 空窗统计
    gaps = []
    prev_end = runs[0][0] + runs[0][3]
    for r in runs[1:]:
        g = r[0] - prev_end
        gaps.append(max(0.0, g))
        prev_end = r[0] + r[3]
    big = [g for g in gaps if g > 1000.0]
    print("\n=== 空窗统计 ===")
    print("  run 数 = %d；总空窗 = %.1f ms；>1ms 的空窗 %d 个，合计 %.1f ms" % (
        len(runs), sum(gaps) / 1000, len(big), sum(big) / 1000))
    print("  最大的 5 个空窗(ms)：%s" % ", ".join("%.1f" % (g / 1000) for g in sorted(gaps, reverse=True)[:5]))

    lines = []
    lines.append("# 序号  相对ms    空窗ms   次数   总耗时ms  名字")
    prev_end = runs[0][0]
    for k, (s, n, c, d, e) in enumerate(runs[: args.max_runs]):
        gap = max(0.0, s - prev_end) / 1000
        lines.append("%5d %10.1f %8.1f x%-6d %9.3f  %s" % (k, (s - t0) / 1000, gap, c, d / 1000, n[:70]))
        prev_end = e
    text = "\n".join(lines) + "\n"
    if args.out:
        open(args.out, "w").write(text)
        print("\n已写 %d 个 run 到 %s" % (min(len(runs), args.max_runs), args.out))
    else:
        print("\n" + text[:8000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
