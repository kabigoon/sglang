#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/weights/DeepSeek-R1-0528-w4a8-per-channel}"
PORT="${PORT:-20766}"
INITIAL_PER_DP_CHUNK_SIZE="${INITIAL_PER_DP_CHUNK_SIZE:-2048}"
PP_ASYNC_BATCH_DEPTH="${PP_ASYNC_BATCH_DEPTH:-2}"
DP_SIZE=4
SERVER_CHUNK_BUDGET="$((INITIAL_PER_DP_CHUNK_SIZE * DP_SIZE))"
export SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR="${SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR:-0.75}"

exec sglang serve \
  --model-path "${MODEL_PATH}" \
  --tp-size 4 \
  --pp-size 4 \
  --pp-max-micro-batch-size 16 \
  --pp-async-batch-depth "${PP_ASYNC_BATCH_DEPTH}" \
  --trust-remote-code \
  --attention-backend ascend \
  --device npu \
  --quantization modelslim \
  --dtype bfloat16 \
  --watchdog-timeout 9000 \
  --mem-fraction-static 0.85 \
  --max-running-requests 64 \
  --context-length 8188 \
  --max-total-tokens 100000 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --chunked-prefill-size "${SERVER_CHUNK_BUDGET}" \
  --max-prefill-tokens 4096 \
  --enable-dynamic-chunking \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --enable-dp-attention \
  --dp-size "${DP_SIZE}" \
  --enable-dp-lm-head \
  --reasoning-parser deepseek-r1 \
  --tool-call-parser deepseekv3 \
  --host 127.0.0.1 \
  --port "${PORT}"
