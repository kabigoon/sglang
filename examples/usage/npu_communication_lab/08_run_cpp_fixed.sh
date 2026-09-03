#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/weights/DeepSeek-R1-0528-w4a8-per-channel}"
PORT="${PORT:-20766}"
PER_DP_CHUNK_SIZE="${PER_DP_CHUNK_SIZE:-1024}"
PP_ASYNC_BATCH_DEPTH="${PP_ASYNC_BATCH_DEPTH:-0}"
DP_SIZE=4
SERVER_CHUNK_BUDGET="$((PER_DP_CHUNK_SIZE * DP_SIZE))"

# Total workers = TP per stage (4) * PP stages (4) = 16 NPUs.
# Speculative decoding is intentionally absent: current SGLang PP rejects it.
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
  --max-prefill-tokens 2048 \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --enable-dp-attention \
  --dp-size "${DP_SIZE}" \
  --enable-dp-lm-head \
  --reasoning-parser deepseek-r1 \
  --tool-call-parser deepseekv3 \
  --host 127.0.0.1 \
  --port "${PORT}"
