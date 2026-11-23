#!/usr/bin/env bash
# dp=4 起 server 的调试脚本（单独起 server，不带 bench）。
# 与 oneshot 的区别：日志同时落盘、崩了直接给针对性提示、参数可逐个试。
#
# 用法：
#   bash _tmp_preflight/run_dp4_debug.sh                  # 默认 dp=4 + decode 图开（贴近最终配置）
#   DECODE_GRAPH=0 bash _tmp_preflight/run_dp4_debug.sh   # 图开不起来时的退路
#   DRY_RUN=1 bash _tmp_preflight/run_dp4_debug.sh        # 只打印将要用的参数
#
# 起来后另开终端跑 client：
#   WARMUP=4 NUM_PROMPTS=8 CONCURRENCY=1 DP_SIZE=4 CACHE_PCT=50 bash _tmp_preflight/bench_128k_1k.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)

DP_SIZE=${DP_SIZE:-4}
CHUNKED_PREFILL=${CHUNKED_PREFILL:-$((4096 * DP_SIZE))}
MEM_FRACTION=${MEM_FRACTION:-0.85}
DECODE_GRAPH=${DECODE_GRAPH:-1}
DECODE_BS=${DECODE_BS:-"1 2 4 8 16"}
LOG=${LOG:-/tmp/dp${DP_SIZE}_$(date +%m%d_%H%M%S).log}
DRY_RUN=${DRY_RUN:-0}

cat <<EOF
==================== dp=${DP_SIZE} 起 server ====================
  tp / dp        : 16 / ${DP_SIZE}   ⇒ 每 DP 组内 attn_tp = $((16 / DP_SIZE))
  chunked prefill: ${CHUNKED_PREFILL} 全局 / $((CHUNKED_PREFILL / DP_SIZE)) 每 rank
  mem-fraction   : ${MEM_FRACTION}
  decode graph   : ${DECODE_GRAPH}   bs=[${DECODE_BS}]
  spec           : 开（DSpark）
  日志           : ${LOG}
==================================================================
EOF

if [ "$DRY_RUN" = "1" ]; then
  echo "[DRY] 将执行："
  echo "  DP_SIZE=$DP_SIZE CHUNKED_PREFILL=$CHUNKED_PREFILL MEM_FRACTION=$MEM_FRACTION \\"
  echo "  DECODE_GRAPH=$DECODE_GRAPH DECODE_BS=\"$DECODE_BS\" bash $HERE/run_dsv41_w4a8.sh"
  exit 0
fi

export DP_SIZE CHUNKED_PREFILL MEM_FRACTION DECODE_GRAPH DECODE_BS
bash "$HERE/run_dsv41_w4a8.sh" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}

echo
echo "==================== 退出码 $rc | 日志 $LOG ===================="
if [ "$rc" != "0" ]; then
  echo "--- 关键报错 ---"
  grep -E "capture_bs|Loaded weights leave no GPU|post_torch|block_tables|OutOfMemory|NPU out of memory" "$LOG" | tail -6
  echo
  echo "--- 对号入座 ---"
  cat <<'HINT'
  capture_bs=[]                      → decode 图抓不了：改成 DECODE_GRAPH=0
                                       或再放宽 bs：DECODE_BS="1 2 4 8 16 32 64"
  Loaded weights leave no GPU memory → 权重+池子撑爆预算：MEM_FRACTION=0.88
  post_torch / NPU out of memory     → 激活不够：CHUNKED_PREFILL=8192（2048/rank）
  block_tables                       → 开了 prefill 图，DSV4.1 不支持：去掉 PREFILL_GRAPH
  其它                               → tail -40 上面日志路径，把最后 20 行发我
HINT
else
  echo "server 正常退出（通常是你 Ctrl-C 停的）"
fi
