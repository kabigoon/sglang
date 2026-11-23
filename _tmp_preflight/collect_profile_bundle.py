#!/usr/bin/env python3
"""collect_profile_bundle.py —— 从 torch_npu profiler 产物里抽出**小体积**证据包，便于 git 推送。

为什么需要它：一个 rank 的 ASCEND_PROFILER_OUTPUT 动辄上百 MB，但真正要看的信息
（按 stream 聚合、按算子聚合、top kernel、每步耗时）加起来只有几百 KB。

用法：
  python3 _tmp_preflight/collect_profile_bundle.py /tmp/prof_b1 --tag prof_b1
  python3 _tmp_preflight/collect_profile_bundle.py /tmp/prof_128k --tag prof_128k --max-kb 400

产物目录（git 可推）：
  _tmp_preflight/profile_bundle/<tag>/
    README.txt            本包是什么、怎么生成的
    meta.txt              机器/时间/git sha/源目录
    stream_summary.csv    每个 Stream ID：kernel 数、总耗时、占比、top 算子  ← 认泳道用
    step_trace_time.csv   每步耗时（原样复制）
    op_statistic_top.csv  算子聚合 top N
    kernel_top.csv        耗时 top N 的 kernel（含 pipe 比率列）
    comm_summary.csv      HCCL 通信汇总
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict

BUNDLE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile_bundle")

KERNEL_FIELDS = [
    "Stream ID", "Name", "Type", "Accelerator Core", "Duration(us)",
    "aicore_time(us)", "aic_mac_ratio", "aic_mte1_ratio", "aic_mte2_ratio",
    "aic_fixpipe_ratio", "aiv_vec_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio",
    "cube_utilization(%)", "OP State",
]


def find_rank_dir(root: str) -> str:
    cands = sorted(glob.glob(os.path.join(root, "*_ascend_pt")))
    if not cands:
        if os.path.isdir(os.path.join(root, "ASCEND_PROFILER_OUTPUT")):
            return root
        sys.exit("没找到 *_ascend_pt 目录：%s" % root)
    if len(cands) > 1:
        print("[warn] 找到 %d 个 rank 目录，只取第一个：%s" % (len(cands), cands[0]))
    return cands[0]


def read_csv_rows(path: str) -> tuple[list[str], list[dict]]:
    with open(path, newline="", errors="replace") as f:
        rd = csv.DictReader(f)
        return list(rd.fieldnames or []), list(rd)


def to_float(v, default=0.0) -> float:
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return default


def build_stream_summary(kernel_csv: str, out_csv: str) -> int:
    if not os.path.isfile(kernel_csv):
        print("[warn] 没有 kernel_details.csv，跳过 stream 汇总")
        return 0
    _, rows = read_csv_rows(kernel_csv)
    per = defaultdict(lambda: {"n": 0, "dur": 0.0, "ops": defaultdict(float), "cores": defaultdict(float)})
    for r in rows:
        sid = (r.get("Stream ID") or "?").strip()
        dur = to_float(r.get("Duration(us)"))
        name = (r.get("Name") or "?").strip()
        core = (r.get("Accelerator Core") or "?").strip()
        d = per[sid]
        d["n"] += 1
        d["dur"] += dur
        d["ops"][name] += dur
        d["cores"][core] += dur

    total = sum(v["dur"] for v in per.values()) or 1.0
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Stream ID", "kernels", "total_us", "pct_of_all", "top_core",
                    "top_ops(us)"])
        for sid, v in sorted(per.items(), key=lambda kv: -kv[1]["dur"]):
            top_ops = "; ".join("%s=%.0f" % (n, d)
                                for n, d in sorted(v["ops"].items(), key=lambda kv: -kv[1])[:4])
            top_core = max(v["cores"].items(), key=lambda kv: kv[1])[0]
            w.writerow([sid, v["n"], "%.1f" % v["dur"], "%.2f%%" % (100.0 * v["dur"] / total),
                        top_core, top_ops])
    print("  stream_summary.csv: %d 条 stream" % len(per))
    return len(per)


def build_kernel_top(kernel_csv: str, out_csv: str, top: int) -> int:
    if not os.path.isfile(kernel_csv):
        return 0
    fields, rows = read_csv_rows(kernel_csv)
    cols = [c for c in KERNEL_FIELDS if c in fields]
    rows.sort(key=lambda r: -to_float(r.get("Duration(us)")))
    keep = rows[:top]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in keep:
            w.writerow([r.get(c, "") for c in cols])
    print("  kernel_top.csv: %d 行（按 Duration 排序）" % len(keep))
    return len(keep)


def build_op_top(op_csv: str, out_csv: str, top: int) -> int:
    if not os.path.isfile(op_csv):
        print("[warn] 没有 op_statistic.csv，跳过")
        return 0
    fields, rows = read_csv_rows(op_csv)
    rows.sort(key=lambda r: -to_float(r.get("Total Time(us)")))
    keep = rows[:top]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in keep:
            w.writerow(r)
    print("  op_statistic_top.csv: %d 行" % len(keep))
    return len(keep)


def build_comm_summary(comm_json: str, out_csv: str, top: int) -> int:
    """communication_matrix.json 结构随版本变化，这里做尽力而为的通用汇总。"""
    if not os.path.isfile(comm_json):
        return 0
    try:
        data = json.load(open(comm_json, errors="replace"))
    except Exception as e:
        print("[warn] communication_matrix.json 解析失败: %s" % e)
        return 0

    agg = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)

    def walk(node):
        if isinstance(node, dict):
            # 常见的键名组合，做一个宽松匹配
            name = (node.get("OpType") or node.get("op_type") or node.get("Name")
                    or node.get("name") or node.get("GroupName"))
            dur = node.get("Duration(us)") or node.get("duration") or node.get("ElapseTime")
            size = node.get("TransitSize") or node.get("Size") or node.get("DataSize")
            if name is not None:
                key = str(name)[:60]
                counts[key] += 1
                if dur is not None:
                    agg[key]["dur_us"] += to_float(dur)
                if size is not None:
                    agg[key]["bytes"] += to_float(size)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(data)
    if not agg:
        print("[warn] communication_matrix.json 里没解析出可聚合字段，原样复制")
        shutil.copyfile(comm_json, out_csv.replace(".csv", ".json"))
        return 0

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comm_op", "count", "total_us", "total_bytes"])
        for k, v in sorted(agg.items(), key=lambda kv: -kv[1].get("dur_us", 0.0))[:top]:
            w.writerow([k, counts[k], "%.1f" % v.get("dur_us", 0.0), "%.0f" % v.get("bytes", 0.0)])
    print("  comm_summary.csv: %d 类通信算子" % min(len(agg), top))
    return min(len(agg), top)


def git_sha(repo: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="trace 根目录，例如 /tmp/prof_b1")
    ap.add_argument("--tag", default=None, help="产物子目录名，默认取 root 的 basename")
    ap.add_argument("--top", type=int, default=200, help="top N 行数上限")
    ap.add_argument("--max-kb", type=int, default=400, help="总体积上限（KB），超了只警告")
    args = ap.parse_args()

    tag = args.tag or os.path.basename(os.path.abspath(args.root))
    src_rank = find_rank_dir(args.root)
    ap_dir = os.path.join(src_rank, "ASCEND_PROFILER_OUTPUT")
    dst = os.path.join(BUNDLE_ROOT, tag)

    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)

    print("源：%s" % ap_dir)
    print("目标：%s" % dst)

    # 原样复制的小文件
    for name in ("step_trace_time.csv", "api_statistic.csv", "op_statistic.csv"):
        p = os.path.join(ap_dir, name)
        if os.path.isfile(p):
            shutil.copyfile(p, os.path.join(dst, name))
            print("  复制 %s（%.1f KB）" % (name, os.path.getsize(p) / 1024.0))

    build_stream_summary(os.path.join(ap_dir, "kernel_details.csv"),
                         os.path.join(dst, "stream_summary.csv"))
    build_kernel_top(os.path.join(ap_dir, "kernel_details.csv"),
                     os.path.join(dst, "kernel_top.csv"), args.top)
    build_op_top(os.path.join(ap_dir, "op_statistic.csv"),
                 os.path.join(dst, "op_statistic_top.csv"), args.top)
    build_comm_summary(os.path.join(ap_dir, "communication_matrix.json"),
                       os.path.join(dst, "comm_summary.csv"), 50)

    total = sum(os.path.getsize(os.path.join(dst, f)) for f in os.listdir(dst))
    with open(os.path.join(dst, "meta.txt"), "w") as f:
        f.write("generated_at = %s\n" % subprocess.run(
            ["date", "-Is"], capture_output=True, text=True).stdout.strip())
        f.write("host = %s\n" % os.uname().nodename)
        f.write("git_head = %s\n" % git_sha(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        f.write("source_rank_dir = %s\n" % src_rank)
        f.write("command = %s\n" % " ".join(sys.argv))
        f.write("bundle_bytes = %d\n" % total)

    with open(os.path.join(dst, "README.txt"), "w") as f:
        f.write(__doc__ or "")

    total = sum(os.path.getsize(os.path.join(dst, x)) for x in os.listdir(dst))
    print("\n产物总体积：%.1f KB（上限 %d KB）" % (total / 1024.0, args.max_kb))
    if total > args.max_kb * 1024:
        print("[warn] 超过上限，建议减小 --top 或只挑需要的文件推送")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
