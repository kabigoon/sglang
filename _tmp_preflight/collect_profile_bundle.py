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
    kernel_schema.txt     kernel_details.csv 的列名/分布/样本序列（几 KB，用来离线核对切步假设）
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


def _dir_weight(d: str) -> int:
    """证据包只取一个 rank 目录：按「料多少」排序，别按目录名。

    DP_REPLICA 门控会留下 N 个目录（每个 replica 一个），其中只有**拿到请求**的那个
    有真正的算子和 attention，其余是空闲 rank。按名字取第一个很可能取到空的那个。
    """
    kd = os.path.join(d, "ASCEND_PROFILER_OUTPUT", "kernel_details.csv")
    try:
        return os.path.getsize(kd)
    except OSError:
        total = 0
        for root, _dirs, files in os.walk(d):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return total


def find_rank_dir(root: str) -> str:
    cands = sorted(glob.glob(os.path.join(root, "*_ascend_pt")))
    if not cands:
        if os.path.isdir(os.path.join(root, "ASCEND_PROFILER_OUTPUT")):
            return root
        sys.exit("没找到 *_ascend_pt 目录：%s" % root)
    if len(cands) > 1:
        cands.sort(key=_dir_weight, reverse=True)
        print("[info] 找到 %d 个 rank 目录，按 kernel_details 大小取最有料的一个：" % len(cands))
        for d in cands:
            print("        %9.1f MB  %s" % (_dir_weight(d) / 1e6, os.path.basename(d)))
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


def build_kernel_schema(kernel_csv: str, out_txt: str, sample: int = 60) -> int:
    """把 kernel_details.csv 的「列名 + 分布 + 序列样本」抽成几 KB 的文本。

    为什么需要：analyze_prefill_steps.py 的切步完全依赖这张表的列结构、以及
    「同一 chunk 内 kernel 序列是否重复」。这些只有真 trace 能回答。采 3 步 smoke 时
    把这份 schema 推回来，就能离线把分析器改对，而不是拿 32 步的大 trace 去试错。

    回答四个问题：
      1) 到底有哪些列（Start Time / Model ID / Stream ID 是否都在）；
      2) Model ID 是否可用（取值种类数）；
      3) kernel 名序列的重复周期是「一层」还是「一个 chunk」；
      4) attention 类 kernel 的 duration 序列是否呈「同 chunk 内恒定、跨 chunk 跳变」。
    """
    if not os.path.isfile(kernel_csv):
        print("[warn] 没有 kernel_details.csv，跳过 schema")
        return 0
    fields, rows = read_csv_rows(kernel_csv)
    per_stream = defaultdict(int)
    per_name = defaultdict(lambda: [0, 0.0])
    model_ids = defaultdict(int)
    for r in rows:
        per_stream[(r.get("Stream ID") or "?").strip()] += 1
        nm = (r.get("Name") or "?").strip()
        per_name[nm][0] += 1
        per_name[nm][1] += to_float(r.get("Duration(us)"))
        if "Model ID" in fields:
            model_ids[(r.get("Model ID") or "").strip()] += 1

    def num(r, key):
        try:
            return float(r.get(key))
        except (TypeError, ValueError):
            return None

    with open(out_txt, "w") as f:
        f.write("kernel_details.csv 结构诊断\n")
        f.write("=" * 60 + "\n")
        f.write("行数 = %d\n" % len(rows))
        f.write("列（%d）:\n  %s\n\n" % (len(fields), ", ".join(fields)))
        f.write("Stream ID 分布（前 10）:\n")
        for sid, n in sorted(per_stream.items(), key=lambda kv: -kv[1])[:10]:
            f.write("  %-10s %8d\n" % (sid, n))
        f.write("\nModel ID 取值种类 = %d" % len(model_ids))
        if len(model_ids) <= 8:
            f.write("  → %s" % dict(list(model_ids.items())[:8]))
        f.write("\n\nkernel 名 top %d（次数, 总耗时ms）:\n" % min(20, len(per_name)))
        for nm, (c, us) in sorted(per_name.items(), key=lambda kv: -kv[1][0])[:20]:
            f.write("  %-8d %12.1f  %s\n" % (c, us / 1000.0, nm[:80]))

        # 主计算流：kernel 数最多的流
        main_sid = max(per_stream.items(), key=lambda kv: kv[1])[0]
        main = [r for r in rows if (r.get("Stream ID") or "").strip() == main_sid]
        f.write("\n主计算流 = %s（%d 个 kernel）\n" % (main_sid, len(main)))
        f.write("该流前 %d 个 kernel（Start Time(us), Duration(us), Name）:\n" % sample)
        for i, r in enumerate(main[:sample]):
            f.write("  %3d  %12s %10s  %s\n" % (
                i, r.get("Start Time(us)"), r.get("Duration(us)"),
                (r.get("Name") or "")[:70]))

        # attention 类 kernel 的 duration 序列：判断「同 chunk 恒定、跨 chunk 跳变」
        import re as _re
        att = [r for r in rows
               if _re.search(r"attention|attn|mla|indexer", (r.get("Name") or "").lower())]
        f.write("\nattention 类 kernel 共 %d 个（前 %d 个的 Start/Duration）:\n" % (
            len(att), min(sample * 2, len(att))))
        for r in att[:sample * 2]:
            f.write("  %12s %10s  %s\n" % (r.get("Start Time(us)"), r.get("Duration(us)"),
                                            (r.get("Name") or "")[:60]))
        f.write("\n注意：若上面 duration 呈「若干连续相等值 + 阶跃」= 周期性(chunk)结构；\n")
        f.write("      若 layer 间 duration 交替变化 = 一层内就有不同窗口长度的层，需要按名字分组。\n")
    print("  kernel_schema.txt: %d 行 kernel 的结构诊断（%.1f KB）" % (
        len(rows), os.path.getsize(out_txt) / 1024.0))
    return 1


def build_comm_summary(comm_json: str, out_csv: str, top: int) -> int:
    """解析 communication_matrix.json。

    实测 schema（CANN 9.2 / torch_npu 2.13）：
      {"step": {"p2p": {...},
                "collective": {"allreduce-top1@<id>": {"0-0": {"Transport Type": "LOCAL",
                                                              "Transit Size(MB)": 25.08,
                                                              "Transit Time(ms)": 0.0423,
                                                              "Bandwidth(GB/s)": 579.0,
                                                              "Op Name": "hcom_allReduce__503_9_1"}, ...},
                               "allreduce-middle@<id>": {...}}}}
    注意 "Transit Time(ms)" 是**每条链路**的传输时间，各链路并行，所以整体时长更接近 max 而非 sum。
    """
    if not os.path.isfile(comm_json):
        return 0
    try:
        data = json.load(open(comm_json, errors="replace"))
    except Exception as e:
        print("[warn] communication_matrix.json 解析失败: %s" % e)
        return 0

    coll = ((data.get("step") or {}).get("collective")) or {}
    if not coll:
        print("[warn] 没有 step.collective，原样复制 json")
        shutil.copyfile(comm_json, out_csv.replace(".csv", ".json"))
        return 0

    rows = []
    for name, links in coll.items():
        if not isinstance(links, dict):
            continue
        n = 0
        times, sizes, bws = [], [], []
        transports = defaultdict(int)
        ops = set()
        for _k, m in links.items():
            if not isinstance(m, dict):
                continue
            n += 1
            t = to_float(m.get("Transit Time(ms)"))
            if t:
                times.append(t)
            sizes.append(to_float(m.get("Transit Size(MB)")))
            b = to_float(m.get("Bandwidth(GB/s)"))
            if b:
                bws.append(b)
            transports[(m.get("Transport Type") or "?").strip()] += 1
            op = m.get("Op Name")
            if op:
                ops.add(str(op))
        if not n:
            continue
        rows.append([
            re.sub(r"@\d+$", "", str(name)),          # 去掉尾部 @id，便于同类聚合
            n,
            "%.4f" % (max(times) if times else 0.0),
            "%.4f" % sum(times),
            "%.2f" % sum(sizes),
            "%.1f" % (sum(bws) / len(bws) if bws else 0.0),
            ",".join("%s:%d" % kv for kv in sorted(transports.items(), key=lambda kv: -kv[1])),
            len(ops),
        ])
    if not rows:
        print("[warn] collective 里没有可解析链路，原样复制 json")
        shutil.copyfile(comm_json, out_csv.replace(".csv", ".json"))
        return 0

    rows.sort(key=lambda r: -float(r[3]))
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["collective", "links", "max_link_ms", "sum_link_ms", "total_MB",
                    "avg_GBps", "transports", "distinct_ops"])
        w.writerows(rows[:top])
    print("  comm_summary.csv: %d 类通信（按 sum_link_ms 排序）" % min(len(rows), top))
    return min(len(rows), top)


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
    build_kernel_schema(os.path.join(ap_dir, "kernel_details.csv"),
                        os.path.join(dst, "kernel_schema.txt"))
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
