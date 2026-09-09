#!/bin/bash
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0

source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# A5 DSpark must use the custom_transformer built from this vllm-ascend tree.
# Do not append the inherited ASCEND_CUSTOM_OPP_PATH here: it may contain the
# incompatible system custom_transformer under /usr/local/Ascend.
VLLM_ASCEND_ROOT=/home/a00821909/vllm-ascend
VLLM_CUSTOM_VENDOR="${VLLM_ASCEND_ROOT}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
VLLM_CUSTOM_OPAPI_LIB="${VLLM_CUSTOM_VENDOR}/op_api/lib"
VLLM_ASCEND_BINDING="${VLLM_ASCEND_ROOT}/build/vllm_ascend_C.cpython-312-x86_64-linux-gnu.so"

export ASCEND_CUSTOM_OPP_PATH="${VLLM_CUSTOM_VENDOR}:/usr/local/Ascend/cann-9.1.0/opp/vendors/customize"
export LD_LIBRARY_PATH="${VLLM_CUSTOM_OPAPI_LIB}:${VLLM_ASCEND_ROOT}/build:${LD_LIBRARY_PATH}"
export SGLANG_DSPARK_A5_EXTRA_OPS_SO="${VLLM_ASCEND_BINDING}"

VLLM_CUSTOM_OPMASTER="${VLLM_CUSTOM_VENDOR}/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/libcust_opmaster_rt2.0.so"
if grep -aFq 'oriSparseIndices is not supported now' "${VLLM_CUSTOM_OPMASTER}"; then
    echo "ERROR: selected vllm-ascend tiling library is the old build: ${VLLM_CUSTOM_OPMASTER}" >&2
    exit 1
fi
echo "A5 DSpark binding: ${SGLANG_DSPARK_A5_EXTRA_OPS_SO}"
echo "A5 DSpark custom OPP: ${ASCEND_CUSTOM_OPP_PATH}"


export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_SET_CPU_AFFINITY=1

#TEST
export TASK_QUEUE_ENABLE=1
export INF_NAN_MODE_FORCE_DISABLE=1
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=max

#HCCL deepep
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export HCCL_BUFFSIZE=1024
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

# 蚂蚁搬家，ROUND*TOKENS≥chunkedprefillsize/tp*dp
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
export DEEPEP_NORMAL_LONG_SEQ_ROUND=16
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=2048


# dsv4
export IS_DEEPSEEK_V4=1 
export USE_FUSED_HC_PRE_ASCENDC=1
export SGLANG_DSV4_NPU_FUSED_COMPRESSOR=1
export SGLANG_DSV4_NPU_FUSED_COMPRESSOR_PREFILL=1


# skip gpu branch
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export FORCE_DRAFT_MODEL_NON_QUANT=1
export SGLANG_DSV4_FP4_EXPERTS=True
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_OPT_FP8_WO_A_GEMM=False

export PYTHONPATH=/home/l00567497/sglang/python:$PYTHONPATH
export MODEL_PATH=/home/weights/DeepSeek-V4-Flash-0731

# PLOG
export COLLECT_LOGS_PATH=/home/a00821909/plog  # 设置用于收集日志的环境路径变量
rm -rf "$COLLECT_LOGS_PATH"
mkdir "$COLLECT_LOGS_PATH"
export ASCEND_SLOG_PRINT_TO_STDOUT=0  # 1/0 Plog是否打屏（推荐为0）
export ASCEND_GLOBAL_LOG_LEVEL=3  # 日志等级 0: debug 1: info 2: warning 3: error
export ASCEND_PROCESS_LOG_PATH="$COLLECT_LOGS_PATH"  # 设置Plog存储路径

# perfermance
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_NPU_USE_MULTI_STREAM=1
# export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1
# export SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=100
export USE_NPU_MOE_GATING_TOP_K=1
# export SGLANG_DP_USE_REDUCE_SCATTER=1
# export ASCEND_LAUNCH_BLOCKING=1



export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=max

# Dspark
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_DSPARK_FAST_KERNEL=0

python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
    --page-size 128 \
    --tp-size 4 \
    --trust-remote-code \
    --attention-backend dsv4 \
    --device npu \
    --watchdog-timeout 9000 \
    --host 0.0.0.0 --port 8001 \
    --mem-fraction-static 0.72 \
    --max-running-requests 64 \
    --chunked-prefill-size 131072 \
    --max-prefill-tokens 131072 \
    --cuda-graph-bs 4 8 10 16\
    --kv-cache-dtype auto \
    --enable-dp-lm-head \
    --disable-radix-cache  \
    --enable-dp-attention --dp-size 4  \
    --reasoning-parser deepseek-v4 \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path "${MODEL_PATH}" \
    --speculative-draft-model-quantization modelslim \
    --speculative-draft-attention-backend ascend \
    --speculative-num-draft-tokens 6 \
    --speculative-dspark-block-size 5 \
    --moe-a2a-backend deepep \
    --deepep-mode auto 

    # --disable-overlap-schedule
    # --json-model-override-args "{\"num_hidden_layers\":4}" \
    # --disable-cuda-graph \
    # --disable-overlap-schedule \
    # --chunked-prefill-size -1
    # --disable-cuda-graph \
    # --moe-a2a-backend deepep --deepep-mode normal \
    # --enable-dp-lm-head \
    # -skip-server-warmup 
    # # --ep-size 2
    # --cuda-graph-backend-decode disable



