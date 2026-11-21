#!/usr/bin/env bash
# GSM8K 精度快测（evalscope, openai_api）。默认 200 条、贪心解码。
#
# 用法：
#   bash _tmp_preflight/eval_gsm8k.sh                  # 默认打本机 5684 端口，200 条
#   LIMIT=50 bash _tmp_preflight/eval_gsm8k.sh         # 先抽 50 条试水
#   API_URL=http://127.0.0.1:5684/v1 DATASET_DIR=/path/to/gsm8k bash _tmp_preflight/eval_gsm8k.sh
#
# 依赖：evalscope 已安装；服务已 ready；本地 gsm8k 数据集目录存在。
set -u

# ---------------- 可覆盖参数 ----------------
API_URL=${API_URL:-http://127.0.0.1:5684/v1}
API_KEY=${API_KEY:-EMPTY}
MODEL_NAME=${MODEL_NAME:-deepseek-v4.1-flash-a3}   # openai_api 模式只是个标签，随便写
LIMIT=${LIMIT:-200}
MAX_TOKENS=${MAX_TOKENS:-8192}        # 千万别用 250000：会超过上下文、且几乎必然超时
TEMPERATURE=${TEMPERATURE:-0.0}       # 精度评测用贪心，别用 0.9
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}
TIMEOUT=${TIMEOUT:-1800}
ENABLE_THINKING=${ENABLE_THINKING:-true}
DATASET_DIR=${DATASET_DIR:-$PWD/gsm8k}

RUN_TIME="$(date +%Y%m%d_%H%M%S)"
LOG="EVAL_${RUN_TIME}.log"
WORK_DIR="evalscope_out_${RUN_TIME}"

echo "===== [0] 参数 ====="
cat <<EOF
  API_URL        = $API_URL
  MODEL_NAME     = $MODEL_NAME
  LIMIT          = $LIMIT
  MAX_TOKENS     = $MAX_TOKENS
  TEMPERATURE    = $TEMPERATURE
  EVAL_BATCH_SIZE= $EVAL_BATCH_SIZE
  DATASET_DIR    = $DATASET_DIR
  LOG            = $LOG
  WORK_DIR       = $WORK_DIR
EOF

# ---------------- 前置检查（省得跑一半才发现） ----------------
echo
echo "===== [1] 前置检查 ====="
FAIL=0

if ! command -v evalscope >/dev/null 2>&1; then
  echo "  FAIL: 找不到 evalscope 命令（pip show evalscope 看看；没装就先 pip install evalscope）"
  FAIL=1
else
  echo "  evalscope: $(command -v evalscope)  $(evalscope --version 2>/dev/null | head -1)"
fi

if [ -d "$DATASET_DIR" ] && [ -n "$(ls -A "$DATASET_DIR" 2>/dev/null)" ]; then
  echo "  dataset: OK  ($(ls -1 "$DATASET_DIR" | head -3 | tr '\n' ' '))"
else
  echo "  FAIL: 数据集目录不存在或为空: $DATASET_DIR"
  echo "        需要放 evalscope 的 gsm8k 本地文件（test-*.parquet 或 *.jsonl），"
  echo "        或用 DATASET_DIR=/绝对路径 指向它。"
  FAIL=1
fi

# 服务可达性：/v1/models
MODELS_CODE=$(curl -s -o /dev/null -m 15 -w '%{http_code}' -H "Authorization: Bearer $API_KEY" "${API_URL%/v1}/v1/models" 2>/dev/null)
[ -z "$MODELS_CODE" ] && MODELS_CODE=000
if [ "$MODELS_CODE" = "200" ]; then
  echo "  server: OK  (GET ${API_URL%/v1}/v1/models -> 200)"
else
  echo "  FAIL: 服务不可达 (HTTP $MODELS_CODE)。确认 launch_server 已 ready、端口/URL 正确。"
  echo "        你之前脚本里是 61.47.19.67:30000，本机 run_dsv41.sh 是 0.0.0.0:5684。"
  FAIL=1
fi

if [ "$FAIL" = "1" ]; then
  echo
  echo "前置检查未通过，先修上面 FAIL 项；参数可覆盖后重跑。"
  exit 1
fi

# ---------------- 正式评测 ----------------
echo
echo "===== [2] 开始评测（$LIMIT 条）====="
GEN_CFG=$(cat <<JSON
{
  "temperature": $TEMPERATURE,
  "max_tokens": $MAX_TOKENS,
  "seed": 3407,
  "n": 1,
  "timeout": $TIMEOUT,
  "stream": true,
  "extra_body": {"chat_template_kwargs": {"enable_thinking": $ENABLE_THINKING}}
}
JSON
)

evalscope eval \
  --model "$MODEL_NAME" \
  --api-url "$API_URL" \
  --api-key "$API_KEY" \
  --eval-type openai_api \
  --generation-config "$GEN_CFG" \
  --datasets gsm8k \
  --dataset-hub Local \
  --dataset-args "{\"gsm8k\": {\"local_path\": \"$DATASET_DIR\", \"shuffle\": true, \"shuffle_choices\": true}}" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --ignore-errors \
  --work-dir "$WORK_DIR" \
  --limit "$LIMIT" \
  2>&1 | tee "$LOG"

# ---------------- 结果摘要 ----------------
echo
echo "===== [3] 结果摘要 ====="
grep -Ei "gsm8k|accuracy|score" "$LOG" | tail -15
echo
echo "结果目录: $(pwd)/$WORK_DIR"
echo "完整日志: $(pwd)/$LOG"

echo
echo "===== 回传块（复制下面几行）====="
echo "LIMIT=$LIMIT MAX_TOKENS=$MAX_TOKENS TEMP=$TEMPERATURE THINKING=$ENABLE_THINKING"
echo "WORK_DIR=$WORK_DIR"
grep -Ei "averageaccuracy|accuracy" "$LOG" | tail -3
echo "===== 回传块结束 ====="
