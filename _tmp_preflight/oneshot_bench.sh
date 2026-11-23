#!/usr/bin/env bash
# 一次性实验：起 server → 等 ready → 跑 bench → 打回传块 → （默认）停 server。
# 机器不独占时用这个：一次一条命令，跑完就释放。
#
# 用法（一次尝试）：
#   DP_SIZE=2 bash _tmp_preflight/oneshot_bench.sh                    # 每 rank 4096 自动推导
#   DP_SIZE=4 bash _tmp_preflight/oneshot_bench.sh
#   DP_SIZE=1 CHUNKED_PREFILL=4096 bash _tmp_preflight/oneshot_bench.sh
#   KEEP_SERVER=1 DP_SIZE=2 bash _tmp_preflight/oneshot_bench.sh       # 跑完保留 server
#
# 标度曲线（同一个 server，只变上下文长度，省掉每次 3-5 分钟的起服务）：
#   DP_SIZE=4 CHUNKED_PREFILL=16384 DECODE_GRAPH=1 DECODE_BS="1 2 4 8 16" \
#     WARMUP=4 NUM_PROMPTS=8 CONCURRENCY=1 CACHE_PCT=50 SCALE="65536 98304 126250" \
#     bash _tmp_preflight/oneshot_bench.sh
#
# 单点指定输入长度（不透传 IN_LEN 时由 bench 脚本推导）：
#   IN_LEN=98304 bash _tmp_preflight/oneshot_bench.sh
#
# 算子开关扫描（每轮只加一个 K=V，且必须过 GSM8K 才认）：
#   EXTRA_ENV="SGLANG_OPT_FUSE_WQA_WKV=1" bash _tmp_preflight/oneshot_bench.sh
#
# 已知：dp != 8 时 decode 图会 AssertionError: capture_bs=[]，本脚本自动 DECODE_GRAPH=0；
#       想保住 decode 图（TPOT 才与 §5.2 可比）显式给 DECODE_GRAPH=1 DECODE_BS="1 2 4 8 16"。
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SEG=$(cd "$HERE/.." && pwd)

DP_SIZE=${DP_SIZE:-8}
CHUNKED_PREFILL=${CHUNKED_PREFILL:-$((4096 * DP_SIZE))}   # 保持每 DP rank 4096
MEM_FRACTION=${MEM_FRACTION:-0.85}
if [ -z "${DECODE_GRAPH:-}" ]; then
  if [ "$DP_SIZE" -eq 8 ]; then DECODE_GRAPH=1; else DECODE_GRAPH=0; fi
fi
DECODE_BS=${DECODE_BS:-}
EXTRA_ENV=${EXTRA_ENV:-}
DISABLE_SPEC=${DISABLE_SPEC:-0}
NUM_PROMPTS=${NUM_PROMPTS:-8}
WARMUP=${WARMUP:-$DP_SIZE}          # 覆盖所有 DP rank 的前缀缓存
CACHE_PCT=${CACHE_PCT:-50}          # 命中率固定 50%，不改
CONCURRENCY=${CONCURRENCY:-1}
IN_LEN=${IN_LEN:-}                  # 空=让 bench 脚本自己推导（默认 128K 自适应）
OUT_LEN=${OUT_LEN:-}
SKIP_FLUSH=${SKIP_FLUSH:-}
SCALE=${SCALE:-}                    # 非空=同一 server 依次跑这些名义输入长度
KEEP_SERVER=${KEEP_SERVER:-0}
DRY_RUN=${DRY_RUN:-0}

OUT=${OUT:-/tmp/oneshot_$(date +%m%d_%H%M%S)_dp${DP_SIZE}}
mkdir -p "$OUT"
SLOG="$OUT/server.log"

echo "===== 本次尝试 ====="
echo "  dp=$DP_SIZE  chunk=${CHUNKED_PREFILL}(全局)/$((CHUNKED_PREFILL / DP_SIZE))(每rank)  mem=$MEM_FRACTION"
echo "  decode_graph=$DECODE_GRAPH  bs=[${DECODE_BS:-默认}]  spec=$([ "$DISABLE_SPEC" = "1" ] && echo off || echo on)"
echo "  CACHE_PCT=$CACHE_PCT  warmup=$WARMUP  prompts=$NUM_PROMPTS  conc=$CONCURRENCY"
[ -n "$SCALE" ] && echo "  标度曲线 SCALE=[$SCALE]（同一个 server，只变上下文长度）"
[ -n "$IN_LEN" ] && echo "  IN_LEN=$IN_LEN"
[ -n "$EXTRA_ENV" ] && echo "  EXTRA_ENV=$EXTRA_ENV   ← 本轮唯一变量"
echo "  输出目录: $OUT"
echo

if [ "$DRY_RUN" = "1" ]; then
  echo "[DRY] 会执行："
  echo "  env DP_SIZE=$DP_SIZE CHUNKED_PREFILL=$CHUNKED_PREFILL MEM_FRACTION=$MEM_FRACTION DECODE_GRAPH=$DECODE_GRAPH bash $HERE/run_dsv41_w4a8.sh"
  echo "  env WARMUP=$WARMUP NUM_PROMPTS=$NUM_PROMPTS CONCURRENCY=$CONCURRENCY DP_SIZE=$DP_SIZE CACHE_PCT=$CACHE_PCT IN_LEN=${IN_LEN:-<auto>} bash $HERE/bench_128k_1k.sh"
  if [ -n "$SCALE" ]; then
    for L in $SCALE; do
      echo "  env WARMUP=$WARMUP NUM_PROMPTS=$NUM_PROMPTS CONCURRENCY=$CONCURRENCY DP_SIZE=$DP_SIZE CACHE_PCT=$CACHE_PCT IN_LEN=$L bash $HERE/bench_128k_1k.sh"
    done
  fi
  exit 0
fi

cleanup() { pkill -9 -f "sglang.launch_server" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "[1/3] 停旧 server"
cleanup; sleep 3

echo "[2/3] 起 server（日志 $SLOG）"
( cd "$SEG" && env DP_SIZE="$DP_SIZE" CHUNKED_PREFILL="$CHUNKED_PREFILL" \
    MEM_FRACTION="$MEM_FRACTION" DECODE_GRAPH="$DECODE_GRAPH" \
    ${DECODE_BS:+DECODE_BS="$DECODE_BS"} \
    ${EXTRA_ENV:+EXTRA_ENV="$EXTRA_ENV"} \
    ${DISABLE_SPEC:+DISABLE_SPEC="$DISABLE_SPEC"} \
    bash "$HERE/run_dsv41_w4a8.sh" ) > "$SLOG" 2>&1 &

wait_s=0
until curl --noproxy '*' -fsS -m 5 http://127.0.0.1:5684/v1/models >/dev/null 2>&1; do
  if ! pgrep -f "sglang.launch_server" >/dev/null 2>&1; then
    echo "  [FAIL] server 进程已退出，看日志：" && tail -20 "$SLOG"; exit 1
  fi
  sleep 10; wait_s=$((wait_s + 10))
  if [ "$wait_s" -ge 900 ]; then echo "  [FAIL] 等待 900s 仍未 ready" && tail -20 "$SLOG"; exit 1; fi
done
echo "  ready（等待 ${wait_s}s）"
grep -oE "max_total_num_tokens=[0-9]+|chunked_prefill_size=[0-9]+|available_gpu_mem=[0-9.]+ GB" "$SLOG" | head -3 | sed 's/^/    /'
echo "    chunked prefill: ${CHUNKED_PREFILL}(全局) -> $((CHUNKED_PREFILL / DP_SIZE)) 每 DP rank"

if [ "$DECODE_GRAPH" = "0" ] && [ "$DP_SIZE" -ne 8 ]; then
  echo "    [提醒] decode 图已关 → TPOT 与 §5.2（graph on，32.8ms）不可比；"
  echo "           要对齐请加 DECODE_GRAPH=1 DECODE_BS=\"1 2 4 8 16\""
fi

# 本轮要跑的输入长度序列：SCALE 优先，其次 IN_LEN，最后交给 bench 自己推导
if [ -n "$SCALE" ]; then
  POINTS="$SCALE"
  echo
  echo "[3/3] 标度曲线：依次跑 IN_LEN = $POINTS（server 配置不变，唯一变量=上下文长度）"
else
  POINTS="${IN_LEN:--}"
fi

TSV="$OUT/summary.tsv"
: > "$TSV"
RC_ALL=0
for L in $POINTS; do
  if [ "$L" = "-" ]; then TAG=auto; BENCH_IN_LEN=""; else TAG="$L"; BENCH_IN_LEN="$L"; fi
  BLOG="$OUT/bench_in${TAG}.log"
  echo
  echo "----- 点 IN_LEN=$TAG -----"
  env WARMUP="$WARMUP" NUM_PROMPTS="$NUM_PROMPTS" CONCURRENCY="$CONCURRENCY" \
      DP_SIZE="$DP_SIZE" CACHE_PCT="$CACHE_PCT" \
      ${BENCH_IN_LEN:+IN_LEN="$BENCH_IN_LEN"} \
      ${OUT_LEN:+OUT_LEN="$OUT_LEN"} \
      ${SKIP_FLUSH:+SKIP_FLUSH="$SKIP_FLUSH"} \
      bash "$HERE/bench_128k_1k.sh" 2>&1 | tee "$BLOG" | tail -30
  RC=${PIPESTATUS[0]}
  [ "$RC" -ne 0 ] && RC_ALL=$RC
  # 抽数进汇总（读 bench 落盘的 jsonl，比 grep 文本稳）
  python3 - "$BLOG" "$TAG" "$TSV" <<'PY'
import json, re, sys

log, tag, tsv = sys.argv[1], sys.argv[2], sys.argv[3]
row = [tag, "0", "-", "-", "-", "-", "-", "-", "-", "-"]
try:
    txt = open(log, errors="replace").read()
    paths = re.findall(r"结果文件: (\S+\.jsonl)", txt)
    rows = [l for l in open(paths[-1]) if l.strip()]
    r = json.loads(rows[-1])
    cr = r.get("cache_report") or {}
    n = max(len(r.get("input_lens") or [0]), 1)
    prompt = cr.get("total_prompt_tokens") or 0
    cached = cr.get("total_cached_tokens") or 0
    actual_in = sum(r.get("input_lens") or [0]) / n
    unc = (prompt - cached) / n if prompt else float("nan")
    ttft_mean = r.get("ttft_mean_ms", float("nan")) / 1000.0
    ttft_med = r.get("ttft_median_ms", float("nan")) / 1000.0
    tpot = r.get("tpot_mean_ms", float("nan"))
    tput = r.get("output_throughput", float("nan"))
    hit = cr.get("cache_hit_rate_pct", float("nan"))
    rate = unc / ttft_med if ttft_med else float("nan")
    row = [tag, str(n), "%.0f" % actual_in, "%.2f" % hit, "%.0f" % unc,
           "%.1f" % ttft_mean, "%.1f" % ttft_med, "%.1f" % tpot,
           "%.1f" % tput, "%.1f" % rate]
except Exception as e:  # 失败也占一行，别让表格错位
    row[9] = "ERR"
    sys.stderr.write("[warn] 抽数失败 %s: %s\n" % (tag, e))
open(tsv, "a").write("\t".join(row) + "\n")
PY
done

echo
echo "===== 结果（复制这段） ====="
grep -hE "^HIT=|^TTFT_|^TPOT_|^OUT_TPUT=" "$OUT"/bench_in*.log 2>/dev/null | tail -8
echo
python3 - "$TSV" "$DP_SIZE" "$((CHUNKED_PREFILL / DP_SIZE))" "$MEM_FRACTION" "$DECODE_GRAPH" "$CACHE_PCT" "$SCALE" <<'PY'
import math, sys, time

tsv, dp, chunk, mem, dg, pct, scale = sys.argv[1:8]
rows = [(l.rstrip("\n").split("\t") + ["-"] * 10)[:10] for l in open(tsv)]
title = "标度曲线" if scale else "单点结果"
print("===== %s（复制整段回传） =====" % title)
print("dp=%s chunk/rank=%s mem=%s decode_graph=%s spec=on cache=%s%%  %s"
      % (dp, chunk, mem, dg, pct, time.strftime("%F")))
print("%-9s %5s %10s %6s %8s %11s %10s %7s %9s %8s"
      % ("IN_LEN", "reqs", "actual_in", "HIT%", "U_unc", "TTFT_mean", "TTFT_med", "TPOT", "TPUT", "U/s"))
for c in rows:
    print("%-9s %5s %10s %6s %8s %11s %10s %7s %9s %8s"
          % (c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]))

def num(x):
    return float(x)

good = []
for c in rows:
    try:
        if c[2] != "-" and c[6] != "-" and float(c[2]) > 0 and float(c[6]) > 0:
            good.append(c)
    except ValueError:
        pass

if len(good) >= 2:
    print()
    print("标度指数（用 actual_in 与实际 TTFT_med，检验平滑幂律 vs 路径突变）：")
    for a, b in zip(good, good[1:]):
        T1, T2, t1, t2 = num(a[2]), num(b[2]), num(a[6]), num(b[6])
        print("  %s -> %s : T x%.2f  TTFT x%.2f  =>  指数 %.2f"
              % (a[0], b[0], T2 / T1, t2 / t1, math.log(t2 / t1) / math.log(T2 / T1)))
    print()
    print("判据：指数随长度平滑变化 → 平滑幂律，问题在 kernel 常数因子；")
    print("      某点突跳（该段指数明显高于邻段）→ 那里有路径突变，先查分支/kernel 选择。")

if not scale:
    c = rows[-1]
    try:
        label = "%.0fk/1k" % (float(c[2]) / 1000.0)
    except ValueError:
        label = "?k/1k"
    print()
    print("登记行：")
    print("  %s | cache %s%% | dp=%s chunk/rank=%s mem=%s graph=%s spec=on | HIT=%s%% U=%s "
          "TTFT_mean=%ss TTFT_median=%ss TPOT=%sms TPUT=%s tok/s | %s"
          % (label, pct, dp, chunk, mem, dg, c[3], c[4], c[5], c[6], c[7], c[8],
             time.strftime("%F")))
print("===== 回传块结束 =====")
PY
echo
echo "server log: $SLOG"
echo "汇总 tsv : $TSV"
echo "各点日志 : ${OUT}/bench_in*.log"
echo "（退出码 ${RC_ALL}：0=全部完整性通过）"

if [ "$KEEP_SERVER" = "1" ]; then
  echo
  echo "KEEP_SERVER=1：server 保留运行（pid $(pgrep -f sglang.launch_server | head -1)）"
  echo "  只跑 client：WARMUP=$WARMUP NUM_PROMPTS=$NUM_PROMPTS CONCURRENCY=$CONCURRENCY DP_SIZE=$DP_SIZE CACHE_PCT=$CACHE_PCT IN_LEN=<长度> bash $HERE/bench_128k_1k.sh"
  trap - EXIT INT TERM
else
  echo
  echo "释放机器：停 server"
fi
