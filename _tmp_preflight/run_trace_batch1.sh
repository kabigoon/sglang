#!/usr/bin/env bash
# 批次① 打点启动器：只做三件事 —— source vendor 环境、导出 trace env、跑你自己的 run_dsv41.sh。
# 不需要改 run_dsv41.sh（prefill 本来就是 eager，关不关图都不影响 prefill 打点）。
# 用法： bash _tmp_preflight/run_trace_batch1.sh
# 可选： RANKS=0,1 LIMIT=1 MAXNEW=8 bash _tmp_preflight/run_trace_batch1.sh
set -u
cd "$(dirname "$0")/.." || exit 1

for f in /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash \
         /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash; do
  if [ -f "$f" ]; then
    set +u; . "$f" >/dev/null 2>&1; set -u
    echo "sourced: $f"
  fi
done

DIR=${DIR:-/tmp/dsv41_trace}
rm -rf "$DIR"
export SGLANG_DSV41_TRACE=1
export SGLANG_DSV41_TRACE_DIR="$DIR"
export SGLANG_DSV41_TRACE_RANKS="${RANKS:-0}"
# 0 = 不限：warmup、图捕获、你的请求都会记；靠 tail/看 T 区分是哪一个
export SGLANG_DSV41_TRACE_LIMIT="${LIMIT:-0}"
unset SGLANG_DSV41_TRACE_LAYERS 2>/dev/null || true

cat <<EOF

================ 批次① 打点已开启 ================
  TRACE_DIR = $DIR
  RANKS     = $SGLANG_DSV41_TRACE_RANKS
  LIMIT     = $SGLANG_DSV41_TRACE_LIMIT   (0=不限)
  tag       = model_in / model_out

起服务、看到 "The server is fired up and ready to roll!" 之后，
发 【一个】 请求（max_new_tokens 给小一点，省得日志太长）：

  curl -s http://127.0.0.1:5684/generate -H 'Content-Type: application/json' \\
    -d '{"text":"你好，请用一句话介绍你自己","sampling_params":{"max_new_tokens":${MAXNEW:-8},"temperature":0}}' \\
    | head -c 400

然后回传这两个切片：

  head -4 $DIR/r00.log
  grep "model_in\\|model_out" $DIR/r00.log | tail -4; echo "---"; grep -c model_in $DIR/r00.log

=================================================
EOF

exec bash ./run_dsv41.sh
