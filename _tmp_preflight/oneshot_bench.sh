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
# 已知：dp != 8 时 decode 图会 AssertionError: capture_bs=[]，本脚本自动 DECODE_GRAPH=0。
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SEG=$(cd "$HERE/.." && pwd)

DP_SIZE=${DP_SIZE:-8}
CHUNKED_PREFILL=${CHUNKED_PREFILL:-$((4096 * DP_SIZE))}   # 保持每 DP rank 4096
MEM_FRACTION=${MEM_FRACTION:-0.85}
if [ -z "${DECODE_GRAPH:-}" ]; then
  if [ "$DP_SIZE" -eq 8 ]; then DECODE_GRAPH=1; else DECODE_GRAPH=0; fi
fi
NUM_PROMPTS=${NUM_PROMPTS:-8}
WARMUP=${WARMUP:-$DP_SIZE}          # 覆盖所有 DP rank 的前缀缓存
CACHE_PCT=${CACHE_PCT:-50}          # 命中率固定 50%，不改
CONCURRENCY=${CONCURRENCY:-1}
KEEP_SERVER=${KEEP_SERVER:-0}
DRY_RUN=${DRY_RUN:-0}

OUT=${OUT:-/tmp/oneshot_$(date +%m%d_%H%M%S)_dp${DP_SIZE}}
mkdir -p "$OUT"
SLOG="$OUT/server.log"
BLOG="$OUT/bench.log"

echo "===== 本次尝试 ====="
echo "  dp=$DP_SIZE  chunk=${CHUNKED_PREFILL}(全局)/$((CHUNKED_PREFILL / DP_SIZE))(每rank)  mem=$MEM_FRACTION"
echo "  decode_graph=$DECODE_GRAPH  spec=on  CACHE_PCT=$CACHE_PCT  warmup=$WARMUP  prompts=$NUM_PROMPTS"
echo "  输出目录: $OUT"
echo

if [ "$DRY_RUN" = "1" ]; then
  echo "[DRY] 会执行："
  echo "  env DP_SIZE=$DP_SIZE CHUNKED_PREFILL=$CHUNKED_PREFILL MEM_FRACTION=$MEM_FRACTION DECODE_GRAPH=$DECODE_GRAPH bash $HERE/run_dsv41_w4a8.sh"
  echo "  env WARMUP=$WARMUP NUM_PROMPTS=$NUM_PROMPTS CONCURRENCY=$CONCURRENCY DP_SIZE=$DP_SIZE CACHE_PCT=$CACHE_PCT bash $HERE/bench_128k_1k.sh"
  exit 0
fi

cleanup() { pkill -9 -f "sglang.launch_server" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "[1/3] 停旧 server"
cleanup; sleep 3

echo "[2/3] 起 server（日志 $SLOG）"
( cd "$SEG" && env DP_SIZE="$DP_SIZE" CHUNKED_PREFILL="$CHUNKED_PREFILL" \
    MEM_FRACTION="$MEM_FRACTION" DECODE_GRAPH="$DECODE_GRAPH" \
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

echo
echo "[3/3] 跑 bench"
env WARMUP="$WARMUP" NUM_PROMPTS="$NUM_PROMPTS" CONCURRENCY="$CONCURRENCY" \
    DP_SIZE="$DP_SIZE" CACHE_PCT="$CACHE_PCT" \
    bash "$HERE/bench_128k_1k.sh" 2>&1 | tee "$BLOG" | tail -25

echo
echo "===== 结果（复制这段） ====="
grep -E "^HIT=|^TTFT_|^TPOT_|^OUT_TPUT=" "$BLOG" | tail -4
echo "server log: $SLOG"
echo "bench  log: $BLOG"

# 一行式登记行（直接粘进性能表格）
H=$(grep -o 'HIT=[^ ]*' "$BLOG" | tail -1 | cut -d= -f2)
T1=$(grep -o 'TTFT_mean=[0-9.]*' "$BLOG" | tail -1 | cut -d= -f2)
T2=$(grep -o 'TPOT_mean=[0-9.]*' "$BLOG" | tail -1 | cut -d= -f2)
TP=$(grep -o 'OUT_TPUT=[0-9.]*' "$BLOG" | tail -1 | cut -d= -f2)
echo
echo "===== 登记行（粘到性能表格） ====="
printf '128k/1k | cache %s%% | dp=%s chunk/rank=%s mem=%s spec=on | HIT=%s TTFT=%.0fms TPOT=%.1fms TPUT=%.1f tok/s | %s\n' \
  "$CACHE_PCT" "$DP_SIZE" "$((CHUNKED_PREFILL / DP_SIZE))" "$MEM_FRACTION" "${H:-?}" "${T1:-0}" "${T2:-0}" "${TP:-0}" "$(date +%F)"

if [ "$KEEP_SERVER" = "1" ]; then
  echo
  echo "KEEP_SERVER=1：server 保留运行（pid $(pgrep -f sglang.launch_server | head -1)）"
  trap - EXIT INT TERM
else
  echo
  echo "释放机器：停 server"
fi
