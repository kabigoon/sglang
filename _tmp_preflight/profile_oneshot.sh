#!/usr/bin/env bash
# profile_oneshot.sh —— 一条命令跑完「采集 128K prefill 的 profiling」全流程：
#
#   起 server（带 rank0-only + simplify）→ 等 ready → 预热共享前缀（WARMUP 条）
#   → /start_profile → 只发 1 条 128K 请求 → /stop_profile（含导出解析）
#   → 生成两样东西：
#       ① git 用的小证据包  _tmp_preflight/profile_bundle/<TAG>/   （几十 KB，可 push）
#       ② Insight 用的压缩包 /tmp/<TAG>_insight.tgz                 （拷到 Windows 解压后喂 Insight）
#   → 停 server（KEEP_SERVER=1 可保留）
#
# 用法（机器不独占 / 会掉线时用 nohup）：
#   nohup bash _tmp_preflight/profile_oneshot.sh > /tmp/prof_run.log 2>&1 &
#   tail -f /tmp/prof_run.log
#
# 常用覆盖：
#   DP_SIZE=8 CHUNKED_PREFILL=32768 MEM_FRACTION=0.85 NUM_STEPS=16 TAG=prof_128k \
#     nohup bash _tmp_preflight/profile_oneshot.sh > /tmp/prof_run.log 2>&1 &
#   IN_LEN=98304 TAG=prof_96k ...            # 换个上下文长度
#   KEEP_SERVER=1 ...                        # 采完保留 server
#   HB=30 ...                                # 心跳间隔秒
#
# 预期耗时：起 server ~5min + 预热 8 条 ~10min + 采集请求 ~90s + 导出解析 ~2min ≈ 18 分钟
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SEG=$(cd "$HERE/.." && pwd)

# ---------------- 可覆盖参数 ----------------
DP_SIZE=${DP_SIZE:-8}
CHUNKED_PREFILL=${CHUNKED_PREFILL:-$((4096 * DP_SIZE))}   # 保持每 rank 4096
MEM_FRACTION=${MEM_FRACTION:-0.85}                        # 开 DSpark 必须 >0.817
DECODE_GRAPH=${DECODE_GRAPH:-1}
DECODE_BS=${DECODE_BS:-}
CACHE_PCT=${CACHE_PCT:-50}
OUT_LEN=${OUT_LEN:-8}                    # 输出越短，trace 里 decode 越少
WARMUP=${WARMUP:-$DP_SIZE}               # 前缀缓存按 rank 分区，必须覆盖所有 rank
IN_LEN=${IN_LEN:-}                       # 空=bench 自己推导（128K 自动适配）
NUM_STEPS=${NUM_STEPS:-16}               # = prefill chunk 数(~15) + 少量 decode
RANK0_ONLY=${RANK0_ONLY:-1}
SIMPLIFY=${SIMPLIFY:-1}
TAG=${TAG:-prof_128k}
HB=${HB:-30}
KEEP_SERVER=${KEEP_SERVER:-0}
TP_SIZE=${TP_SIZE:-16}
EXTRA_ARGS=${EXTRA_ARGS:-}       # 透传给启动器的额外 CLI 参数（如 --moe-a2a-backend deepep）
DRY_RUN=${DRY_RUN:-0}
PORT=${PORT:-5684}

OUT=${OUT:-/tmp/prof_oneshot_$(date +%m%d_%H%M%S)}
mkdir -p "$OUT"
SLOG="$OUT/server.log"
PROF_DIR=${PROF_DIR:-/tmp/${TAG}_$(date +%m%d_%H%M%S)}
STAGE="$OUT/stage.txt"
stage() { printf '%s | %s\n' "$(date '+%F %T')" "$*" >> "$STAGE" 2>/dev/null || true; }

echo "===== profiling 采集（128K prefill 单请求）====="
echo "  脚本版本: $(git -C "$SEG" rev-parse --short HEAD 2>/dev/null || echo 未知)"
echo "  dp=${DP_SIZE}  chunk=${CHUNKED_PREFILL}(全局)/$((CHUNKED_PREFILL / DP_SIZE))(每rank)  mem=${MEM_FRACTION}"
echo "  cache=${CACHE_PCT}%  warmup=${WARMUP}  out_len=${OUT_LEN}  num_steps=${NUM_STEPS}"
echo "  rank0_only=${RANK0_ONLY} simplify=${SIMPLIFY}（体积控制）"
[ -n "$EXTRA_ARGS" ] && echo "  extra args: ${EXTRA_ARGS}"
echo "  trace 目录: ${PROF_DIR}"
echo "  工作目录  : ${OUT}"
echo "  证据包 TAG: ${TAG}"
echo

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[DRY] 会依次执行："
  echo "  1) 起 server: DP_SIZE=$DP_SIZE CHUNKED_PREFILL=$CHUNKED_PREFILL MEM_FRACTION=$MEM_FRACTION DECODE_GRAPH=$DECODE_GRAPH EXTRA_ENV=SGLANG_NPU_PROFILE_RANK0_ONLY=$RANK0_ONLY SGLANG_NPU_PROFILE_SIMPLIFY=$SIMPLIFY"
  echo "  2) 预热: WARMUP=$WARMUP NUM_PROMPTS=1 CACHE_PCT=$CACHE_PCT OUT_LEN=$OUT_LEN bench_128k_1k.sh"
  echo "  3) POST /start_profile {output_dir:$PROF_DIR, num_steps:$NUM_STEPS, activities:[CPU,GPU], with_stack:false}"
  echo "  4) 采集: WARMUP=0 SKIP_FLUSH=1 NUM_PROMPTS=1 CACHE_PCT=$CACHE_PCT OUT_LEN=$OUT_LEN bench_128k_1k.sh"
  echo "  5) POST /stop_profile"
  echo "  6) collect_profile_bundle.py $PROF_DIR --tag $TAG"
  echo "  7) tar czf /tmp/${TAG}_insight.tgz -C $(dirname "$PROF_DIR") $(basename "$PROF_DIR")"
  exit 0
fi

# ---------------- 工具函数 ----------------
cleanup() {
  if [ -n "${LAUNCH_PID:-}" ]; then
    command -v pkill >/dev/null 2>&1 && pkill -9 -P "$LAUNCH_PID" 2>/dev/null || true
    kill -9 "$LAUNCH_PID" 2>/dev/null || true
  fi
  if command -v pkill >/dev/null 2>&1; then
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

ready_probe() {
  python3 - <<'PY'
import http.client
try:
    c = http.client.HTTPConnection("127.0.0.1", 5684, timeout=5)
    c.request("GET", "/v1/models")
    r = c.getresponse(); r.read()
    print(r.status)
except Exception as e:
    print("ERR:%s" % type(e).__name__)
PY
}

api_post() {  # $1=path $2=json $3=timeout
  python3 - "$1" "$2" "${3:-120}" <<'PY'
import http.client, json, sys
path, body, tmo = sys.argv[1], sys.argv[2], int(sys.argv[3])
c = http.client.HTTPConnection("127.0.0.1", 5684, timeout=tmo)
c.request("POST", path, body.encode() or b"{}", {"Content-Type": "application/json"})
print(c.getresponse().read().decode().strip()[:200])
PY
}

# 后台跑 + 心跳（输出进日志，屏幕每 HB 秒一行）
run_with_heartbeat() {  # $1=label $2=logfile，其余=命令
  local label="$1" log="$2"; shift 2
  "$@" > "$log" 2>&1 &
  local pid=$! el=0 last=""
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$HB"; el=$((el + HB))
    last=$(tr '\r' '\n' < "$log" 2>/dev/null | grep -v '^[[:space:]]*$' | tail -1 | cut -c1-90)
    echo "    [${label} ${el}s] ${last}"
  done
  wait "$pid"; local rc=$?
  echo "    ${label} 结束：${el}s rc=${rc}"
  return "$rc"
}

# ---------------- 1. 起 server ----------------
echo "[1/7] 停旧 server 并启动新 server（带 profiling 开关）"
stage "stop old server"
cleanup; sleep 3

PROF_ENV=""
[ "$RANK0_ONLY" = "1" ] && PROF_ENV="SGLANG_NPU_PROFILE_RANK0_ONLY=1"
[ "$SIMPLIFY" = "1" ]   && PROF_ENV="${PROF_ENV:+$PROF_ENV }SGLANG_NPU_PROFILE_SIMPLIFY=1"
echo "    EXTRA_ENV=${PROF_ENV:-（无）}"
stage "launch server dp=$DP_SIZE chunk=$CHUNKED_PREFILL mem=$MEM_FRACTION graph=$DECODE_GRAPH extra=[$PROF_ENV]"

( cd "$SEG" && env DP_SIZE="$DP_SIZE" CHUNKED_PREFILL="$CHUNKED_PREFILL" \
    MEM_FRACTION="$MEM_FRACTION" DECODE_GRAPH="$DECODE_GRAPH" \
    ${DECODE_BS:+DECODE_BS="$DECODE_BS"} \
    ${PROF_ENV:+EXTRA_ENV="$PROF_ENV"} \
    ${EXTRA_ARGS:+EXTRA_ARGS="$EXTRA_ARGS"} \
    bash "$HERE/run_dsv41_w4a8.sh" ) > "$SLOG" 2>&1 &
LAUNCH_PID=$!
disown "$LAUNCH_PID" 2>/dev/null || true
echo "    启动器 pid=${LAUNCH_PID}（前导阶段几秒到几十秒后才出现 python 进程）"

RES=""; wait_s=0; last_rep=0; launch_alive=1
until RES=$(ready_probe) && [ "$RES" = "200" ]; do
  launch_alive=1; kill -0 "$LAUNCH_PID" 2>/dev/null || launch_alive=0
  if [ "$launch_alive" = "0" ] && ! pgrep -f "sglang.launch_server" >/dev/null 2>&1; then
    echo "  [FAIL] 启动器已退出且没有 server 进程，日志尾部："; tail -30 "$SLOG"
    stage "FAIL launcher exited"; exit 1
  fi
  sleep 5; wait_s=$((wait_s + 5))
  if [ $((wait_s - last_rep)) -ge 30 ]; then
    echo "    [${wait_s}s] 等 ready…探测=${RES} 启动器=$([ "$launch_alive" = "1" ] && echo 活 || echo 已退出) 日志末行: $(tail -1 "$SLOG" 2>/dev/null | tr -d '\r' | cut -c1-80)"
    last_rep=$wait_s
  fi
  if [ "$wait_s" -ge 900 ]; then
    echo "  [FAIL] 900s 仍未 ready（探测=${RES}）"; tail -30 "$SLOG"; stage "FAIL ready timeout"; exit 1
  fi
done
echo "  ✓ server ready（${wait_s}s）"
stage "ready after ${wait_s}s"
grep -oE "max_total_num_tokens=[0-9]+" "$SLOG" | tail -1 | sed 's/^/    /'
echo "    启动配置（核对 chunk/mem/attn_tp）："
grep -E "^  (tp-size|mem-fraction|decode graph|chunked prefill|extra env)" "$SLOG" | sed 's/^/      /'

# ---------------- 2. 预热共享前缀 ----------------
echo
echo "[2/7] 预热：${WARMUP} 条（覆盖 ${DP_SIZE} 个 DP rank，让前缀进缓存）"
stage "warmup ${WARMUP} requests"
if ! run_with_heartbeat "预热" "$OUT/warmup.log" \
    env WARMUP="$WARMUP" NUM_PROMPTS=1 CONCURRENCY=1 DP_SIZE="$DP_SIZE" \
        CACHE_PCT="$CACHE_PCT" OUT_LEN="$OUT_LEN" ${IN_LEN:+IN_LEN="$IN_LEN"} \
        bash "$HERE/bench_128k_1k.sh"; then
  echo "  [FAIL] 预热失败，日志：$OUT/warmup.log"; tail -30 "$OUT/warmup.log"; stage "FAIL warmup"; exit 1
fi
grep -hE "^HIT=|^TTFT_" "$OUT/warmup.log" | tail -2 | sed 's/^/    /'

# ---------------- 3. 开 profile ----------------
echo
echo "[3/7] 开 profile（num_steps=${NUM_STEPS}，activities=[CPU,GPU]，with_stack=false）"
stage "start_profile num_steps=$NUM_STEPS"
api_post "/start_profile" "{\"output_dir\":\"${PROF_DIR}\",\"num_steps\":${NUM_STEPS},\"activities\":[\"CPU\",\"GPU\"],\"with_stack\":false}" 60
sleep 2
N_START=$(grep -c "Profiling starts" "$SLOG" 2>/dev/null || echo 0)
N_SKIP=$(grep -c "NPU profiling skipped" "$SLOG" 2>/dev/null || echo 0)
echo "    日志里 Profiling starts=${N_START}（期望 1），skipped=${N_SKIP}（期望 $((TP_SIZE - 1))）"
if [ "${N_START:-0}" -eq 0 ]; then
  echo "  [FAIL] /start_profile 没有生效（日志里没有 'Profiling starts'）"; stage "FAIL start_profile"; exit 1
fi
grep "NPU profiler config" "$SLOG" | head -1 | sed 's/^/    /'

# ---------------- 4. 发采集请求 ----------------
echo
echo "[4/7] 发 1 条 128K/50% 请求（SKIP_FLUSH=1：复用刚灌好的前缀缓存）"
stage "profiled request (1 prompt, out_len=$OUT_LEN)"
if ! run_with_heartbeat "采集" "$OUT/bench.log" \
    env WARMUP=0 SKIP_FLUSH=1 NUM_PROMPTS=1 CONCURRENCY=1 DP_SIZE="$DP_SIZE" \
        CACHE_PCT="$CACHE_PCT" OUT_LEN="$OUT_LEN" ${IN_LEN:+IN_LEN="$IN_LEN"} \
        bash "$HERE/bench_128k_1k.sh"; then
  echo "  [WARN] 采集请求返回非 0（可能是 num_steps 提前触发的正常现象），继续看 stop 结果"
fi
grep -hE "^HIT=|^TTFT_|^OUT_TPUT=" "$OUT/bench.log" | tail -3 | sed 's/^/    /'

# ---------------- 5. 停 profile ----------------
echo
echo "[5/7] 停 profile（导出+解析，128K 可能要 1~3 分钟）"
stage "stop_profile"
api_post "/stop_profile" "{}" 900
for _ in $(seq 1 60); do
  grep -q "Profiling done" "$SLOG" 2>/dev/null && break
  sleep 5
done
if grep -q "Profiling done" "$SLOG" 2>/dev/null; then
  echo "  ✓ Profiling done"
  stage "profiling done"
else
  echo "  [WARN] 等不到 'Profiling done'，可能仍在解析；继续尝试收包"
  stage "WARN profiling done not seen"
fi

# ---------------- 6. 收包 ----------------
echo
echo "[6/7] 收包：① git 证据包  ② Insight 压缩包"
stage "collect bundle"
if [ ! -d "$PROF_DIR" ]; then
  echo "  [FAIL] trace 目录不存在：${PROF_DIR}"; tail -40 "$SLOG"; stage "FAIL no trace dir"; exit 1
fi
du -sh "$PROF_DIR" | sed 's/^/    trace 目录: /'
ls -d "$PROF_DIR"/*_ascend_pt 2>/dev/null | wc -l | sed 's/^/    rank 目录数: /'

python3 "$HERE/collect_profile_bundle.py" "$PROF_DIR" --tag "$TAG" | sed 's/^/    /'
BUNDLE_DIR="$HERE/profile_bundle/$TAG"
du -sh "$BUNDLE_DIR" 2>/dev/null | sed 's/^/    证据包: /'

INSIGHT_TGZ="/tmp/${TAG}_insight.tgz"
tar czf "$INSIGHT_TGZ" -C "$(dirname "$PROF_DIR")" "$(basename "$PROF_DIR")" 2>/dev/null
echo "    Insight 包: ${INSIGHT_TGZ}  ($(du -h "$INSIGHT_TGZ" 2>/dev/null | cut -f1))"
stage "collected bundle=$(du -sh "$BUNDLE_DIR" 2>/dev/null | cut -f1) tgz=$(du -h "$INSIGHT_TGZ" 2>/dev/null | cut -f1)"

# ---------------- 7. 汇总 ----------------
echo
echo "===== 完成（复制这段回传）====="
echo "TAG=${TAG}  dp=${DP_SIZE} chunk/rank=$((CHUNKED_PREFILL / DP_SIZE)) mem=${MEM_FRACTION} cache=${CACHE_PCT}%"
echo "num_steps=${NUM_STEPS} rank0_only=${RANK0_ONLY} simplify=${SIMPLIFY}"
grep -hE "^HIT=|^TTFT_med|^TTFT_median|^TPOT_" "$OUT/bench.log" 2>/dev/null | tail -3
echo "trace 目录 : ${PROF_DIR}  ($(du -sh "$PROF_DIR" 2>/dev/null | cut -f1))"
echo "证据包     : ${BUNDLE_DIR}  ($(du -sh "$BUNDLE_DIR" 2>/dev/null | cut -f1))"
echo "Insight 包 : ${INSIGHT_TGZ}  ($(du -h "$INSIGHT_TGZ" 2>/dev/null | cut -f1))"
echo
echo "下一步（手动）："
echo "  # ① 推送证据包到远端"
echo "  git add _tmp_preflight/profile_bundle/${TAG}"
echo "  git commit -m \"chore(preflight): profile bundle ${TAG} (not part of the PR)\""
echo "  git push origin feat/dsv41-a3-trace"
echo "  # ② 把 Insight 包拷到 Windows（然后解压，用 Insight 打开 *_ascend_pt 目录）"
echo "  #    ${INSIGHT_TGZ}"
echo "===== 回传结束 ====="
stage "done"

if [ "$KEEP_SERVER" = "1" ]; then
  echo
  echo "KEEP_SERVER=1：server 保留运行（pid $(pgrep -f sglang.launch_server | head -1)）"
  trap - EXIT INT TERM
else
  echo
  echo "释放机器：停 server"
  cleanup; sleep 2
  if pgrep -f "sglang.launch_server" >/dev/null 2>&1; then
    echo "  [!] 仍有残留进程，请手动 pkill -9 -f sglang.launch_server"
  else
    echo "  ✓ 已确认停掉"
  fi
  stage "server released"
  trap - EXIT INT TERM
fi
