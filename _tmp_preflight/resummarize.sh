#!/usr/bin/env bash
# resummarize.sh —— 用**已经落盘**的 bench 日志/jsonl 重新打标度表。
# 不启 server、不占卡、不跑请求，秒级完成。用于「跑完了但表格抽数出 nan」这种情况。
#
# 用法：
#   bash _tmp_preflight/resummarize.sh                        # 自动找 /tmp/oneshot_*/bench_in*.log
#   bash _tmp_preflight/resummarize.sh /tmp/oneshot_*/bench_in98304.log
#   bash _tmp_preflight/resummarize.sh /home/dsv4_test/bench_out/v41_....jsonl   # 直接给 jsonl
#
# 注意：sglang 的 jsonl 字段是 "<统计量>_<指标>_ms"（mean_ttft_ms / median_ttft_ms），
#       本脚本两种拼法都认，并在字段缺失时给出可读告警而不是静默 nan。
set -u
HERE=$(cd "$(dirname "$0")" && pwd)

if [ "$#" -eq 0 ]; then
  # shellcheck disable=SC2206
  set -- $(ls -t /tmp/oneshot_*/bench_in*.log 2>/dev/null)
fi
if [ "$#" -eq 0 ]; then
  echo "没找到 bench 日志。/tmp/oneshot_*/bench_in*.log 不存在时，直接把 jsonl 路径传进来："
  echo "  bash _tmp_preflight/resummarize.sh /home/dsv4_test/bench_out/<xxx>.jsonl"
  exit 2
fi

python3 - "$@" <<'PY'
import json, math, os, re, statistics, sys

def row_from_jsonl(path):
    rows = [l for l in open(path) if l.strip()]
    if not rows:
        raise ValueError("空文件: " + path)
    return json.loads(rows[-1])

def find_jsonl(log_path):
    txt = open(log_path, errors="replace").read()
    m = re.findall(r"结果文件: (\S+\.jsonl)", txt)
    if not m:
        raise ValueError("日志里没有『结果文件: …jsonl』这一行")
    p = m[-1]
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p

def first_num(obj, names):
    for nm in names:
        v = obj.get(nm)
        if isinstance(v, (int, float)) and v == v:
            return float(v)
    return float("nan")

def summarize(r):
    cr = r.get("cache_report") or {}
    n = max(len(r.get("input_lens") or [0]), 1)
    prompt = cr.get("total_prompt_tokens") or 0
    cached = cr.get("total_cached_tokens") or 0
    actual_in = sum(r.get("input_lens") or [0]) / n
    unc = (prompt - cached) / n if prompt else float("nan")
    ttft_mean = first_num(r, ("mean_ttft_ms", "ttft_mean_ms"))
    ttft_med = first_num(r, ("median_ttft_ms", "ttft_median_ms"))
    tpot = first_num(r, ("mean_tpot_ms", "tpot_mean_ms"))
    if ttft_med != ttft_med and isinstance(r.get("ttfts"), list):
        ts = [t for t in r["ttfts"] if isinstance(t, (int, float))]
        if ts:
            ttft_mean, ttft_med = statistics.mean(ts), statistics.median(ts)
    if tpot != tpot and isinstance(r.get("itls"), list):
        flat = []
        for x in r["itls"]:
            if isinstance(x, (int, float)):
                flat.append(x)
            elif isinstance(x, list):
                flat += [t for t in x if isinstance(t, (int, float))]
        if flat:
            tpot = statistics.mean(flat)
    if ttft_med != ttft_med:
        sys.stderr.write("[warn] %s: 找不到 TTFT 字段；jsonl 现有键 = %s\n"
                         % (os.path.basename(paths_used[-1]), sorted(r.keys())[:30]))
    hit = cr.get("cache_hit_rate_pct", float("nan"))
    tput = r.get("output_throughput", float("nan"))
    return {
        "n": n, "actual_in": actual_in, "hit": hit, "unc": unc,
        "ttft_mean": ttft_mean / 1000.0, "ttft_med": ttft_med / 1000.0,
        "tpot": tpot, "tput": tput,
        "rate": (unc / (ttft_med / 1000.0)) if ttft_med == ttft_med and ttft_med else float("nan"),
    }

paths_used = []
rows = []
for a in sys.argv[1:]:
    try:
        if a.endswith(".jsonl"):
            jl, tag = a, re.sub(r".*_(\d{6})\.jsonl$", r"\1", os.path.basename(a))
        else:
            jl = find_jsonl(a)
            mm = re.search(r"bench_in(\w+)\.log", os.path.basename(a))
            tag = mm.group(1) if mm else os.path.basename(a)
        paths_used.append(jl)
        r = summarize(row_from_jsonl(jl))
        r["tag"] = tag
        rows.append(r)
        print("  %-9s <- %s" % (tag, jl), file=sys.stderr)
    except Exception as e:
        print("[warn] 跳过 %s: %s: %s" % (a, type(e).__name__, e), file=sys.stderr)

if not rows:
    print("没有任何可解析的结果。", file=sys.stderr)
    sys.exit(1)

print()
print("===== 标度表（从已落盘结果重新生成） =====")
print("%-9s %5s %10s %6s %8s %11s %10s %7s %9s %8s"
      % ("IN_LEN", "reqs", "actual_in", "HIT%", "U_unc", "TTFT_mean", "TTFT_med", "TPOT", "TPUT", "U/s"))
for r in rows:
    def f(v, spec):
        return "-" if v != v else spec % v
    print("%-9s %5d %10.0f %6.2f %8.0f %11s %10s %7s %9s %8s"
          % (r["tag"], r["n"], r["actual_in"], r["hit"], r["unc"],
             f(r["ttft_mean"], "%.1f"), f(r["ttft_med"], "%.1f"),
             f(r["tpot"], "%.1f"), f(r["tput"], "%.1f"), f(r["rate"], "%.1f")))

good = [r for r in rows if r["ttft_med"] == r["ttft_med"] and r["actual_in"] > 0]
if len(good) >= 2:
    print()
    print("标度指数（用 actual_in 与 TTFT_med）：")
    for a, b in zip(good, good[1:]):
        print("  %s -> %s : T x%.2f  TTFT x%.2f  =>  指数 %.2f"
              % (a["tag"], b["tag"], b["actual_in"] / a["actual_in"],
                 b["ttft_med"] / a["ttft_med"],
                 math.log(b["ttft_med"] / a["ttft_med"]) / math.log(b["actual_in"] / a["actual_in"])))

if len(rows) == 1 and rows[0]["ttft_med"] == rows[0]["ttft_med"]:
    r = rows[0]

    def g(v, spec):
        return "?" if v != v else spec % v
    print()
    print("登记行：")
    print("  %.0fk/1k | cache %.2f%% | HIT=%.2f%% U=%.0f TTFT_mean=%ss TTFT_median=%ss TPOT=%sms TPUT=%s tok/s"
          % (r["actual_in"] / 1000.0, r["hit"], r["hit"], r["unc"],
             g(r["ttft_mean"], "%.1f"), g(r["ttft_med"], "%.1f"),
             g(r["tpot"], "%.1f"), g(r["tput"], "%.1f")))
print("===== 结束 =====")
PY
