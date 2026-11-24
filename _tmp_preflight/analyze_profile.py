#!/usr/bin/env python3
"""analyze_profile.py —— 就地聚合 torch_npu profiler 产物，秒级出小表（不用传 20GB trace）。

用法：
  python3 _tmp_preflight/analyze_profile.py /tmp/prof_learn
  python3 _tmp_preflight/analyze_profile.py /tmp/prof_learn --top 25
  python3 _tmp_preflight/analyze_profile.py /tmp/prof_learn --rank 86951     # 只看某个 pid

读什么：
  <root>/<host>_<pid>_<ts>_ascend_pt/ASCEND_PROFILER_OUTPUT/op_statistic.csv
  <root>/<host>_<pid>_<ts>_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv
  <root>/<host>_<pid>_<ts>_ascend_pt/ASCEND_PROFILER_OUTPUT/step_trace_time.csv（有就顺带看）

列（实测）：
  op_statistic:     Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)
  kernel_details:   Device_id,Model ID,Task ID,Stream ID,Name,Type,OP State,Accelerator Core,Start Time(us),
                    Duration(us),Wait Time(us),...,aicore_time(us),...,aic_mac_ratio,...,aiv_vec_ratio,...,
                    cube_utilization(%)

只读，不改任何东西，不占卡。
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict

# ---- 分桶规则：顺序敏感（先匹配到的赢）----
BUCKETS = [
    ("HCCL/通信",   r"allreduce|allgather|reducescatter|reduce_scatter|alltoall|all_to_all|hccl|"
                    r"broadcast|\bsend\b|\brecv\b|dispatch|combine|moe_distribute"),
    ("attention",   r"attention|attn|lightning_indexer|indexer|sharedkv|compress|mla"),
    ("MoE/gating",  r"moe|gating|top_?k|expert|grouped|gmm"),
    ("matmul/linear", r"matmul|gemm|addmm|\bmm\b|batchmatmul|linear|nz_?mm|quant_matmul"),
    ("rope/norm",   r"rotary|rope|rms_?norm|layer_?norm|\bnorm\b|hc_pre|hc_post"),
    ("elementwise", r"add|mul|sub|div|silu|gelu|swiglu|sigmoid|exp|sqrt|rsqrt|abs|neg|pow|log|"
                    r"cast|copy|transpose|contiguous|cumsum|scatter|gather|reshape|slice|fill|zeros|"
                    r"cat|concat|quant|dequant|sort|argsort|argmax|argmin|"
                    r"reduce_?sum|reducesum|reduce_?max|reducemax|reduce_?min|reduce_?mean|"
                    r"clipbyvalue|clip|leftshift|rightshift|\bindex\b|where|select|logical|"
                    r"greater|less|equal|mask|tril|triu|pad|repeat|tile|broadcast|expand|"
                    r"unsqueeze|squeeze|view|stack|split|nonzero|unique"),
]

def bucket_of(op: str) -> str:
    low = op.lower()
    for name, pat in BUCKETS:
        if re.search(pat, low):
            return name
    return "其它"


def _dir_weight(d: str) -> int:
    kd = os.path.join(d, "ASCEND_PROFILER_OUTPUT", "kernel_details.csv")
    try:
        return os.path.getsize(kd)
    except OSError:
        return 0


def rank_dirs(root: str) -> list[str]:
    dirs = sorted(glob.glob(os.path.join(root, "*_ascend_pt")))
    if not dirs:
        # 也允许直接传单个 rank 目录
        if os.path.isdir(os.path.join(root, "ASCEND_PROFILER_OUTPUT")):
            return [root]
    return dirs


def pick_busiest(dirs: list[str]) -> list[str]:
    """多目录时默认只分析「最有料」的那个。

    DP_REPLICA 门控会留下每个 replica 一个目录，只有拿到请求的那个有真算子；
    把空闲 rank 一起相加会把占比稀释成无意义的值（而 --all 仍可显式求和）。
    """
    if len(dirs) <= 1:
        return dirs
    ranked = sorted(dirs, key=_dir_weight, reverse=True)
    print("发现 %d 个 rank 目录，按 kernel_details 大小默认只分析最有料的一个"
          "（要全部相加用 --all）：" % len(dirs))
    for d in ranked:
        print("    %9.1f MB  %s" % (_dir_weight(d) / 1e6, os.path.basename(d)))
    return ranked[:1]


def read_op_stats(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", errors="replace") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "device": (r.get("Device_id") or "").strip(),
                    "op": (r.get("OP Type") or "").strip(),
                    "core": (r.get("Core Type") or "").strip(),
                    "count": int(float(r.get("Count") or 0)),
                    "total_us": float(r.get("Total Time(us)") or 0.0),
                    "ratio": float(r.get("Ratio(%)") or 0.0),
                })
            except (TypeError, ValueError):
                continue
    return rows


def fmt_us(us: float) -> str:
    if us >= 1e6:
        return "%.2fs" % (us / 1e6)
    if us >= 1e3:
        return "%.1fms" % (us / 1e3)
    return "%.0fus" % us


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--rank", default=None, help="只分析 pid 匹配这个子串的 rank")
    ap.add_argument("--per-rank", action="store_true", help="每个 rank 各打一张 top 表（默认只打汇总）")
    ap.add_argument("--all", action="store_true", help="所有 rank 目录相加（默认只取最有料的一个）")
    args = ap.parse_args()

    dirs = rank_dirs(args.root)
    if args.rank:
        dirs = [d for d in dirs if args.rank in os.path.basename(d)]
    elif not args.all:
        dirs = pick_busiest(dirs)
    if not dirs:
        print("没找到 *_ascend_pt 目录（或 --rank 过滤后为空）：%s" % args.root, file=sys.stderr)
        return 2

    print("rank 目录数：%d" % len(dirs))

    # ---- 跨 rank 汇总 ----
    agg_op = defaultdict(lambda: [0, 0.0])          # (bucket, op) -> [count, total_us]
    agg_bucket = defaultdict(lambda: [0, 0.0])      # bucket -> [count, total_us]
    agg_core = defaultdict(lambda: [0, 0.0])        # Core Type -> [count, total_us]
    per_rank_total = {}
    missing = 0

    for d in dirs:
        base = os.path.basename(d)
        pid = re.search(r"_(\d+)_\d{17}_ascend_pt", base)
        pid = pid.group(1) if pid else base
        p = os.path.join(d, "ASCEND_PROFILER_OUTPUT", "op_statistic.csv")
        if not os.path.isfile(p):
            missing += 1
            continue
        rows = read_op_stats(p)
        if not rows:
            missing += 1
            continue
        # 同一个 rank 可能有多个 Device_id，全部计入
        tot = sum(r["total_us"] for r in rows)
        per_rank_total[pid] = tot
        for r in rows:
            b = bucket_of(r["op"])
            agg_op[(b, r["op"])][0] += r["count"]
            agg_op[(b, r["op"])][1] += r["total_us"]
            agg_bucket[b][0] += r["count"]
            agg_bucket[b][1] += r["total_us"]
            core = r["core"] or "?"
            agg_core[core][0] += r["count"]
            agg_core[core][1] += r["total_us"]

    if not per_rank_total:
        print("没有可解析的 op_statistic.csv（missing=%d）" % missing, file=sys.stderr)
        return 1

    grand_total = sum(per_rank_total.values())
    print("解析成功 rank：%d（缺失/空：%d）" % (len(per_rank_total), missing))
    print("所有 rank 的算子耗时总和：%s（注意：这是 %d 个 rank 相加，不是墙钟时间）" % (
        fmt_us(grand_total), len(per_rank_total)))

    # ---- 桶汇总 ----
    print("\n===== 按类别汇总（跨 %d 个 rank 相加） =====" % len(per_rank_total))
    print("%-14s %10s %12s %8s" % ("类别", "调用次数", "总耗时", "占比"))
    for b, (cnt, us) in sorted(agg_bucket.items(), key=lambda kv: -kv[1][1]):
        print("%-14s %10d %12s %7.1f%%" % (b, cnt, fmt_us(us), 100.0 * us / grand_total))

    # ---- Core Type 汇总（AI_CPU 基本就是 host 侧调度出来的算子，值得单独看）----
    print("\n===== 按 Core Type 汇总 =====")
    print("%-12s %10s %12s %8s" % ("Core Type", "调用次数", "总耗时", "占比"))
    for c, (cnt, us) in sorted(agg_core.items(), key=lambda kv: -kv[1][1]):
        print("%-12s %10d %12s %7.1f%%" % (c, cnt, fmt_us(us), 100.0 * us / grand_total))

    # ---- top 算子 ----
    print("\n===== top %d 算子（按总耗时，跨 rank 相加） =====" % args.top)
    print("%-14s %-46s %8s %12s %7s" % ("类别", "OP Type", "次数", "总耗时", "占比"))
    for (b, op), (cnt, us) in sorted(agg_op.items(), key=lambda kv: -kv[1][1])[: args.top]:
        print("%-14s %-46s %8d %12s %6.1f%%" % (b, op[:46], cnt, fmt_us(us), 100.0 * us / grand_total))

    # ---- 每个 rank 的总耗时（看是哪个 rank 特别重/特别闲）----
    print("\n===== 各 rank 算子耗时合计（相对最大值归一） =====")
    mx = max(per_rank_total.values()) or 1.0
    for pid, us in sorted(per_rank_total.items(), key=lambda kv: -kv[1]):
        bar = "#" * int(30 * us / mx)
        print("  pid=%-8s %10s  %5.1f%%  %s" % (pid, fmt_us(us), 100.0 * us / mx, bar))

    # ---- 单 rank 细看 ----
    if args.per_rank:
        d = dirs[0]
        p = os.path.join(d, "ASCEND_PROFILER_OUTPUT", "op_statistic.csv")
        if os.path.isfile(p):
            rows = read_op_stats(p)
            tot = sum(r["total_us"] for r in rows) or 1.0
            print("\n===== 单 rank 明细：%s =====" % os.path.basename(d))
            print("%-14s %-46s %8s %12s %7s" % ("类别", "OP Type", "次数", "总耗时", "占比"))
            for r in sorted(rows, key=lambda r: -r["total_us"])[: args.top]:
                print("%-14s %-46s %8d %12s %6.1f%%"
                      % (bucket_of(r["op"]), r["op"][:46], r["count"], fmt_us(r["total_us"]),
                         100.0 * r["total_us"] / tot))

    # ---- kernel 级 pipe 使用率（有就顺手算）----
    kp = os.path.join(dirs[0], "ASCEND_PROFILER_OUTPUT", "kernel_details.csv")
    if os.path.isfile(kp):
        tot_dur = 0.0
        acc = defaultdict(float)   # ratio 名字 -> 加权和
        nbuf = 0
        with open(kp, newline="", errors="replace") as f:
            rd = csv.DictReader(f)
            ratio_cols = [c for c in (rd.fieldnames or []) if c and re.search(r"_(mac|mte1|mte2|mte3|vec|scalar|fixpipe)_ratio$", c)]
            for r in rd:
                try:
                    dur = float(r.get("Duration(us)") or 0.0)
                except ValueError:
                    continue
                tot_dur += dur
                nbuf += 1
                for c in ratio_cols:
                    v = r.get(c)
                    if v in (None, "", "N/A"):
                        continue
                    try:
                        acc[c] += dur * float(v)
                    except ValueError:
                        pass
        print("\n===== kernel 级 pipe 占用（rank=%s，按 Duration 加权） =====" % os.path.basename(dirs[0]))
        print("  kernel 条数=%d  Duration 合计=%s" % (nbuf, fmt_us(tot_dur)))
        for c, v in sorted(acc.items(), key=lambda kv: -kv[1]):
            if tot_dur:
                print("    %-18s %6.2f%%" % (c, 100.0 * v / tot_dur))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
