#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/weights/DeepSeek-R1-0528-w4a8-per-channel}"
PORT="${PORT:-20766}"
DEEPEP_MODE="${DEEPEP_MODE:-auto}"
PER_DP_CHUNK_SIZE="${PER_DP_CHUNK_SIZE:-256}"
DP_SIZE=16
SERVER_CHUNK_BUDGET="$((PER_DP_CHUNK_SIZE * DP_SIZE))"

exec sglang serve \
  --model-path "${MODEL_PATH}" \
  --tp-size 16 \
  --trust-remote-code \
  --attention-backend ascend \
  --device npu \
  --quantization modelslim \
  --dtype bfloat16 \
  --watchdog-timeout 9000 \
  --mem-fraction-static 0.85 \
  --max-running-requests 256 \
  --context-length 8188 \
  --max-total-tokens 100000 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --chunked-prefill-size "${SERVER_CHUNK_BUDGET}" \
  --max-prefill-tokens 4096 \
  --moe-a2a-backend deepep \
  --deepep-mode "${DEEPEP_MODE}" \
  --enable-dp-attention \
  --dp-size "${DP_SIZE}" \
  --enable-dp-lm-head \
  --reasoning-parser deepseek-r1 \
  --tool-call-parser deepseekv3 \
  --host 127.0.0.1 \
  --port "${PORT}"
