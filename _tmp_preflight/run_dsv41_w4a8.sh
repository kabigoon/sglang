#!/bin/bash
# run_dsv41_w4a8.sh —— 从用户的 run_dsv41.sh 派生，为「128k/1k + 50% 前缀缓存」压测准备。
#
# 相对原脚本的 4 处改动：
#   [1] 权重换成 w4a8（493G vs 728G，TP16 下每 die 省 ~14.7G → KV 池从 46K 涨到数十万 token）
#   [2] 显式打开 modelslim 量化；并把 json-model-override-args 里的 "quantization_config": null 删掉
#       （原来既没 --quantization、又把 quantization_config 抹成 null，很可能按 bf16 加载）
#   [3] 去掉 FORCE_DRAFT_MODEL_NON_QUANT=1（草稿模型不再强制 bf16）
#   [4] 去掉 --disable-radix-cache  ← 否则前缀缓存关闭，「50% cache」永远命中 0%
#
# 用法：
#   bash run_dsv41_w4a8.sh                        # 默认 w4a8 + 开 DSpark + 开前缀缓存
#   VARIANT=w8a8 bash run_dsv41_w4a8.sh           # 换回 w8a8（精度对照用）
#   DISABLE_SPEC=1 bash run_dsv41_w4a8.sh         # 关投机：纯 prefill 基线，TTFT 不受投机影响
#   RADIX_OFF=1 bash run_dsv41_w4a8.sh            # 临时恢复禁用前缀缓存
#
# 注意：本脚本刻意不设 set -u / set -e —— CANN 的 set_env.sh 会引用未定义变量，
#       set -u 会让 source 阶段静默中止整个脚本（表现为"打印完配置就没动静"）。

# ---------------- 可覆盖参数 ----------------
VARIANT=${VARIANT:-w4a8}
DISABLE_SPEC=${DISABLE_SPEC:-0}
RADIX_OFF=${RADIX_OFF:-0}
GRAPH_MAX_SEQ=${GRAPH_MAX_SEQ:-135168}
# 量化方言：modelslim | none | quark ...
# w4a8 若是 Quark/其它格式，用 --quantization modelslim 会报
# "Unsupported Linear modelslim scheme" / "Missing ModelSlim MoE quantization description"
QUANT=${QUANT:-none}
# override 风格：
#   legacy = {"architectures":[...], "quantization_config": null}  ← 你原脚本，w8a8 实测可加载
#   arch   = 只给 architectures（让 config.json 里的 quant_method 生效）
OVERRIDE_STYLE=${OVERRIDE_STYLE:-legacy}
# 原脚本用的是 DeepseekV4ForCausalLM（实测可加载）；checkpoint config 里写的是 V41
ARCH_OVERRIDE=${ARCH_OVERRIDE:-DeepseekV4ForCausalLM}
DRAFT_QUANT=${DRAFT_QUANT:-$QUANT}   # 130048 输入 + 1024 输出 = 131072，留余量
PORT=${PORT:-5684}
TP_SIZE=${TP_SIZE:-16}
DP_SIZE=${DP_SIZE:-8}
# 注意：开 DP attention 时 sglang 会把下面两个全局值除以 attn_dp_size！
#   实际每 DP rank 的 chunked_prefill = CHUNKED_PREFILL / DP_SIZE
# 想要每 rank 4096，就要写 32768（DP=8）。
CHUNKED_PREFILL=${CHUNKED_PREFILL:-4096}
MAX_RUNNING=${MAX_RUNNING:-16}
# ⚠ 这个值决定"权重+KV 池"的总预算，剩下的才是 prefill 激活的余量。
#   池子 4.19M token 对 C≤16 都过剩；prefill 大 chunk 反而需要几 GB 激活内存。
#   128K + chunk 4096 建议 0.85；仍 OOM 再降到 0.80。
MEM_FRACTION=${MEM_FRACTION:-0.96}
# 1 = 开 prefill 图（--cuda-graph-backend-prefill=full）。默认为 0：NPU 上默认被自动禁用。
# 目的：chunk 之间的 CPU launch 开销是 TTFT 的大头，prefill 图可能把它吃掉。
PREFILL_GRAPH=${PREFILL_GRAPH:-0}

case "$VARIANT" in
  w4a8) DEFAULT_W=/home/weights/DeepSeek-V4.1-Flash-1017-w4a8 ;;
  w8a8) DEFAULT_W=/home/weights/DeepSeek-V4.1-Flash-1017-w8a8 ;;
  *) echo "VARIANT 只能是 w4a8 或 w8a8"; exit 2 ;;
esac
export MODEL_PATH=${MODEL_PATH:-$DEFAULT_W}
[ -d "$MODEL_PATH" ] || { echo "权重目录不存在: $MODEL_PATH"; exit 2; }

echo "==================== 启动配置 ===================="
echo "  VARIANT        : $VARIANT"
echo "  MODEL_PATH     : $MODEL_PATH"
echo "  quantization   : ${QUANT}（none=不加 --quantization，靠 checkpoint 描述文件）"
echo "  override 风格  : ${OVERRIDE_STYLE}（legacy=带 quantization_config:null）"
echo "  draft quant    : ${DRAFT_QUANT}"
echo "  DSpark 投机    : $([ "$DISABLE_SPEC" = "1" ] && echo 关闭 || echo 开启)"
echo "  前缀缓存(radix): $([ "$RADIX_OFF" = "1" ] && echo 关闭 || echo 开启)"
echo "  GRAPH_MAX_SEQ  : $GRAPH_MAX_SEQ"
ATTN_TP=$((TP_SIZE / DP_SIZE))
echo "  tp-size / dp   : $TP_SIZE / $DP_SIZE  ⇒ 每 DP 组内 attn_tp = $ATTN_TP"
[ "$ATTN_TP" -eq 1 ] && echo "  [提醒] attn_tp=1：单请求的 attention 只由 1 个 die 算，单请求延迟通常更差（吞吐更好）"
echo "  mem-fraction   : $MEM_FRACTION"
if [ "$DISABLE_SPEC" != "1" ] && awk "BEGIN{exit !($MEM_FRACTION < 0.83)}"; then
  echo "  [warn] 开 DSpark 时 draft 权重也计入预算：MEM_FRACTION 必须 > 0.817（建议 0.85）"
  echo "         否则会报 'Loaded weights leave no GPU memory for the KV cache'"
fi
echo "  prefill graph  : $PREFILL_GRAPH (1=开)"
echo "  chunked prefill: $CHUNKED_PREFILL (全局) -> $((CHUNKED_PREFILL / DP_SIZE)) 每 DP rank"
echo "  max running    : $MAX_RUNNING (全局) -> $((MAX_RUNNING / DP_SIZE)) 每 DP rank"
echo "=================================================="
echo

unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY

echo "[1/4] 停旧进程"
if [ "${SKIP_PKILL:-0}" = "1" ]; then
  echo "      SKIP_PKILL=1，跳过（先自己确认旧 server 已停）"
else
  pkill -9 python 2>/dev/null || true
  pkill -9 sglang 2>/dev/null || true
  sleep 2
  echo "      完成"
fi

echo "[2/4] 准备 JIT 目录"
# JIT 缓存：原脚本用 mktemp 每次新目录（每次重启都重编）。这里默认复用固定目录。
# 想强制重编：JIT_FRESH=1 bash run_dsv41_w4a8.sh
if [ "${JIT_FRESH:-0}" = "1" ]; then
  export TORCH_EXTENSIONS_DIR="$(mktemp -d /tmp/dsv41-jit.XXXXXX)"
else
  export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/dsv41-jit-cache}"
  mkdir -p "$TORCH_EXTENSIONS_DIR"
fi
export CANN_OP_BUILDER_NO_FPRINT_CACHE=1
echo "      TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null 2>&1 || true
sysctl -w vm.swappiness=0 >/dev/null 2>&1 || true
sysctl -w kernel.numa_balancing=0 >/dev/null 2>&1 || true

echo "[3/4] source vendor 环境"
for f in /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash \
         /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash \
         /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
         /usr/local/Ascend/nnal/atb/set_env.sh; do
  if [ -f "$f" ]; then
    echo "      source $f"
    source "$f"
  else
    echo "[warn] 不存在: $f"
  fi
done
echo "      source 完成（ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-未设置}）"

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_SET_CPU_AFFINITY=1

#TEST
export TASK_QUEUE_ENABLE=1
export INF_NAN_MODE_FORCE_DISABLE=1
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=max

export HCCL_BUFFSIZE=128
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
export DEEPEP_NORMAL_LONG_SEQ_ROUND=16
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=2048

# skip gpu branch
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
# [3] 原为 export FORCE_DRAFT_MODEL_NON_QUANT=1  ← 强制草稿模型 bf16，显存杀手，已去掉
export SGLANG_DSV4_FP4_EXPERTS=True
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_OPT_FP8_WO_A_GEMM=False

export PYTHONPATH=/home/a00821909/sglang/python:${PYTHONPATH:-}

# perfermance
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_DSV4_GRAPH_MAX_SEQ=$GRAPH_MAX_SEQ

export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=max

export SGLANG_DSV4_NATIVE_ATTN=0
export SGLANG_SHALLOW_FUSED_ROPE_FQ4=0

export SGLANG_COMPRESSOR_PREFILL_USE_TRITON=1
export SGLANG_COMPRESSOR_DECODE_USE_TRITON=1

export SGLANG_RAGGED_VERIFY_MODE=static

# 组装可选参数
if [ "$OVERRIDE_STYLE" = "legacy" ]; then
  JSON_OVERRIDE="{\"architectures\": [\"$ARCH_OVERRIDE\"], \"quantization_config\": null}"
else
  JSON_OVERRIDE="{\"architectures\": [\"$ARCH_OVERRIDE\"]}"
fi

PGRAPH_ARGS=()
[ "$PREFILL_GRAPH" = "1" ] && PGRAPH_ARGS=(--cuda-graph-backend-prefill=full)

QUANT_ARGS=()
[ "$QUANT" != "none" ] && QUANT_ARGS=(--quantization "$QUANT")

DRAFT_PGRAPH_ARGS=()
[ "$PREFILL_GRAPH" = "1" ] && PGRAPH_ARGS=(--cuda-graph-backend-prefill=full)

QUANT_ARGS=()
[ "$DRAFT_QUANT" != "none" ] && DRAFT_QUANT_ARGS=(--speculative-draft-model-quantization "$DRAFT_QUANT")

SPEC_ARGS=()
if [ "$DISABLE_SPEC" != "1" ]; then
  SPEC_ARGS=(
    --speculative-algorithm DSPARK
    --speculative-draft-model-path "$MODEL_PATH"
    ${DRAFT_QUANT_ARGS[@]+"${DRAFT_QUANT_ARGS[@]}"}
    --speculative-draft-kv-cache-dtype bf16
    --speculative-dspark-block-size 5
  )
fi

RADIX_ARGS=()
[ "$RADIX_OFF" = "1" ] && RADIX_ARGS=(--disable-radix-cache)

# 注意：[2] 去掉了 "quantization_config": null，并显式 --quantization modelslim
echo "[4/4] 启动 server（下面就是 python 原生命令，参数逐一列出）"
python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
    ${QUANT_ARGS[@]+"${QUANT_ARGS[@]}"} \
    --page-size 128 \
    --tp-size ${TP_SIZE} \
    --trust-remote-code \
    --attention-backend dsv4 \
    --device npu \
    --watchdog-timeout 9000 \
    --host 0.0.0.0 --port ${PORT} \
    --mem-fraction-static ${MEM_FRACTION} \
    --max-running-requests ${MAX_RUNNING} \
    --chunked-prefill-size ${CHUNKED_PREFILL} \
    --max-prefill-tokens 131072 \
    --kv-cache-dtype auto \
    --enable-dp-lm-head \
    --cuda-graph-bs-decode 1 2 \
    --enable-dp-attention --dp-size ${DP_SIZE} \
    --reasoning-parser deepseek-v4 \
    --language-model-only \
    --json-model-override-args "${JSON_OVERRIDE}" \
    ${PGRAPH_ARGS[@]+"${PGRAPH_ARGS[@]}"} \
    ${RADIX_ARGS[@]+"${RADIX_ARGS[@]}"} \
    ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"}
