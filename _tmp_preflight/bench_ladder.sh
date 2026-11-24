#!/usr/bin/env bash
# 并发阶梯测试：一次预热，按 CONC_LIST 逐档跑，最后打一张对照表。
#
# 为什么这样省时间：前缀缓存按 DP rank 分区，warmup 要覆盖所有 rank（dp=8 时 8 条 ×
# ~95s ≈ 13 分钟）。本脚本只在第一档做 warmup，后续档用 SKIP_FLUSH=1 复用缓存。
#
# 用法：
#   bash _tmp_preflight/bench_ladder.sh                       # C=1,2,4,8，每档 2*C 条
#   CONC_LIST="1 4" bash _tmp_preflight/bench_ladder.sh
#   NUM_PROMPTS_PER=8 bash _tmp_preflight/bench_ladder.sh      # 每档固定 8 条
set -u

CONC_LIST=${CONC_LIST:-"1 2 4 8"}
WARMUP=${WARMUP:-8}
DP_SIZE=${DP_SIZE:-8}
OUT_DIR=${OUT_DIR:-/tmp/dsv41_ladder}
mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.txt"
: > "$SUMMARY"

echo "并发阶梯：$CONC_LIST | warmup=${WARMUP}（只第一档做）| 结果目录 $OUT_DIR"
echo "预估时长：第 1 档 warmup ~$((WARMUP * 95 / 60)) 分钟 + 各档实测"

FIRST=1
for C in $CONC_LIST; do
  N=${NUM_PROMPTS_PER:-$((C * 2))}
  LOG="$OUT_DIR/c${C}_n${N}.log"
  echo
  echo "==================== C=$C, N=$N ===================="

  if [ "$FIRST" = "1" ]; then
    WARMUP="$WARMUP" NUM_PROMPTS="$N" CONCURRENCY="$C" DP_SIZE="$DP_SIZE" \
      bash "$(dirname "$0")/bench_128k_1k.sh" 2>&1 | tee "$LOG"
    FIRST=0
  else
    SKIP_FLUSH=1 WARMUP=0 NUM_PROMPTS="$N" CONCURRENCY="$C" DP_SIZE="$DP_SIZE" \
      bash "$(dirname "$0")/bench_128k_1k.sh" 2>&1 | tee "$LOG"
  fi

  HIT=$(grep -o 'HIT=[^ ]*' "$LOG" | tail -1 | cut -d= -f2)
  TTFT=$(grep -o 'TTFT_mean=[0-9.]*' "$LOG" | tail -1 | cut -d= -f2)
  TPOT=$(grep -o 'TPOT_mean=[0-9.]*' "$LOG" | tail -1 | cut -d= -f2)
  TPUT=$(grep -o 'OUT_TPUT=[0-9.]*' "$LOG" | tail -1 | cut -d= -f2)
  COMPLETE=$(grep -o 'COMPLETE=[A-Za-z]*' "$LOG" | tail -1 | cut -d= -f2)
  printf "C=%-3s N=%-3s HIT=%-8s TTFT=%12s ms  TPOT=%8s ms  TPUT=%-8s COMPLETE=%s\n" \
    "$C" "$N" "${HIT:-?}" "${TTFT:-?}" "${TPOT:-?}" "${TPUT:-?}" "${COMPLETE:-?}" >> "$SUMMARY"
done

echo
echo "==================== 汇总 ===================="
cat "$SUMMARY"
echo
echo "说明：TTFT 会随并发上升（排队 + MoE 算力争抢）；若要判断\"多少并发下仍满足 TTFT<30s\"，看这张表即可。"
echo "明细日志：$OUT_DIR"
