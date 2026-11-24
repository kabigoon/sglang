#!/usr/bin/env bash
# profile_oneshot.sh —— 一条命令跑完「采集 128K prefill 的 profiling」全流程：
#
#   起 server（带 DP_REPLICA + simplify）→ 等 ready →（默认跳过预热）
#   → /start_profile → 只发 1 条 128K **冷**请求（32 个 chunk）→ /stop_profile（含导出解析）
#   → 生成两样东西：
#       ① git 用的小证据包  _tmp_preflight/profile_bundle/<TAG>/   （几十 KB，可 push）
#       ② Insight 用的压缩包 /tmp/<TAG>_insight.tgz                 （拷到 Windows 解压后喂 Insight）
#   → 停 server（KEEP_SERVER=1 可保留）
#
# 用法（机器不独占 / 会掉线时用 nohup）：
#   nohup bash _tmp_preflight/profile_oneshot.sh > /tmp/prof_run.log 2>&1 &
#   tail -f /tmp/prof_run.log
#
# 常用覆盖（**默认就是冷启动采集**：不预热、一次 32 个 chunk，NUM_STEPS=34）：
#   DP_SIZE=4 CHUNKED_PREFILL=16384 MEM_FRACTION=0.88 DECODE_BS="1 2 4" \
#     EXTRA_ENV="HCCL_BUFFSIZE=128 SGLANG_DSPARK_NVLINK_VOCAB_GATHER=0" \
#     EXTRA_ARGS="--swa-prefix-tails 16" MAX_PREFILL_TOKENS=8192 TAG=prof_128k_cold \
#     nohup bash _tmp_preflight/profile_oneshot.sh > /tmp/prof_run.log 2>&1 &
#
# /start_profile 需要 admin 鉴权：脚本会自己给启动器补 --admin-api-key（ADMIN_KEY 可覆盖），
# 并在 POST 时带 Authorization: Bearer。没有这个会是 403（HANDOFF §15.6）。
#   IN_LEN=98304 TAG=prof_96k ...            # 换个上下文长度
#   KEEP_SERVER=1 ...                        # 采完保留 server
#   HB=30 ...                                # 心跳间隔秒
#
# 两条 env 通道（别搞混）：
#   EXTRA_ENV    K=V，环境变量；会与脚本自己的 profiling 旋鈕**合并**后一起给启动器
#   EXTRA_ARGS   空格分隔的**CLI 参数**，原样传给 launch_server（如 --swa-prefix-tails 16）
#
# rank 门控（DP_REPLICA=1 默认）：每个 attention-DP replica 采一个 rank。
#   dp>1 时**不要**用 RANK0_ONLY=1：C=1 的请求由 round-robin 分到任意 replica，
#   tp_rank 0 很可能整轮 0 token，trace 会显示「attention≈0、Free 很高」而误判方向。
#
# 预期耗时（冷启动）：起 server ~5min + 采集请求 ~110s + 导出解析 2~5min ≈ 12~15 分钟
# 产物：/tmp/<TAG>_<ts>（trace，2~3GB）+ _tmp_preflight/profile_bundle/<TAG>（几十 KB）+ /tmp/<TAG>_insight.tgz
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
# 默认 0 = **冷启动采集**（不预热），理由见 HANDOFF §15.6：
# 原来默认"预热 4 条再采"是错的——warmup 送的是**同一个 prompt**，把它整条灌进缓存后，
# 采集请求 100% 命中、TTFT 1.3s、根本没有 prefill（2026-11-24 实测踩到）。
# 冷启动一次跑 32 个 chunk（prefix 4K→131K），组成和斜率都能拿到，且不依赖缓存状态。
WARMUP=${WARMUP:-0}
IN_LEN=${IN_LEN:-}                       # 空=bench 自己推导（128K 自动适配）
NUM_STEPS=${NUM_STEPS:-34}               # 冷启动 32 chunk(129K/4096) + 2 decode；50% 命中变体只需 20
RANK0_ONLY=${RANK0_ONLY:-0}
# rank 门控（重要）：
#   DP_REPLICA=1（默认）每个 attention-DP replica 采一个 rank（tp_rank % attn_tp == 0）。
#        dp>1 时**必须**用这个：C=1 的请求由 round-robin 分到任意 replica，
#        用 rank0-only 很可能采到「没有 attention 的空闲 replica」，且失败是静默的。
#   RANK0_ONLY=1        只采 tp_rank 0（只在 DP_SIZE=1 时等价于"采到请求的那个 rank"）
DP_REPLICA=${DP_REPLICA:-1}
SIMPLIFY=${SIMPLIFY:-1}
# 体积旋钮（按「压缩力度」从弱到强）：
#   PROF_LEVEL=0     ProfilerLevel.Level0：只留算子级时间线，丢掉 AICore/AIV 的 pipe 明细
#   PROF_NO_AIC=1    不采 AICore PMU（aic_mac/mte/vec 比率全丢）
# ⚠ 这两个会让我们看不到「这个 kernel 是算力受限还是带宽受限」。默认都不用，
#   只有在磁盘实在不够时才降级（先 PROF_LEVEL=0，再 PROF_NO_AIC=1）。
PROF_LEVEL=${PROF_LEVEL:-}
PROF_NO_AIC=${PROF_NO_AIC:-0}
# 生成 Insight 用的 tgz（2-4GB，只为人眼看时间线）。默认关：它会把峰值磁盘占用翻倍。
MAKE_TGZ=${MAKE_TGZ:-0}
# 采集前的最小可用磁盘（GB），不够就早失败，别跑到一半写满盘
DISK_NEED_GB=${DISK_NEED_GB:-5}
TAG=${TAG:-prof_128k}
# /start_profile 与 /stop_profile 是 admin_force（utils/auth_levels.json:70,72）：
# server 必须带 --admin-api-key，POST 必须带 Authorization: Bearer <key>，否则 403。
ADMIN_KEY=${ADMIN_KEY:-sglang-prof-key}
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
echo "  cache=${CACHE_PCT}%  warmup=${WARMUP}(0=冷启动)  out_len=${OUT_LEN}  num_steps=${NUM_STEPS}  admin_key=${ADMIN_KEY}"
echo "  门控: dp_replica=${DP_REPLICA} rank0_only=${RANK0_ONLY} simplify=${SIMPLIFY}（体积控制）"
echo "  外部 env: ${EXTRA_ENV:-（无）}"
[ -n "$EXTRA_ARGS" ] && echo "  extra args: ${EXTRA_ARGS}"
echo "  trace 目录: ${PROF_DIR}"
# 体积预估：smoke 实测 258 MB/step/rank（**simplify 已开**，Level1+pipe），
# 再乘「采集 rank 数」和步数。注意 4 个 replica 的 rank 都在干活（FFN 是 TP16 全 die 参与），
# 所以它们不是"空闲目录"，每个都有真实体积。
NCOLL=1; [ "$DP_REPLICA" = "1" ] && NCOLL=$DP_SIZE
BASE_MB=$(( NUM_STEPS * 258 * NCOLL ))
[ "$SIMPLIFY" != "1" ] && BASE_MB=$(( BASE_MB * 3 ))
[ "$PROF_LEVEL" = "0" ] && BASE_MB=$(( BASE_MB / 2 ))
[ "$PROF_NO_AIC" = "1" ] && BASE_MB=$(( BASE_MB / 2 ))
EST_LO=$(( BASE_MB * 8 / 10 )); EST_HI=$(( BASE_MB * 12 / 10 ))
echo "  预计体积  : $(awk -v a=$EST_LO -v b=$EST_HI 'BEGIN{printf "%.1f~%.1f GB", a/1024, b/1024}')（${NUM_STEPS} step × ${NCOLL} 个采集 rank）"
echo "  压缩比    : csv/json 文本约 10:1（smoke：3.1GB → 326MB），需要 Insight 时再 MAKE_TGZ=1"
echo "  可压缩手段: PROF_LEVEL=0（/2，丢 pipe 明细）、PROF_NO_AIC=1（/2，丢 aic 比率）"
AVAIL_KB=$(df -Pk "$(dirname "$PROF_DIR")" 2>/dev/null | awk 'NR==2{print $4}')
if [ -n "${AVAIL_KB:-}" ]; then
  AVAIL_GB=$(( AVAIL_KB / 1024 / 1024 ))
  echo "  可用磁盘  : ${AVAIL_GB} GB（$(dirname "$PROF_DIR")，要求 ≥ ${DISK_NEED_GB} GB）"
  if [ "$AVAIL_GB" -lt "$DISK_NEED_GB" ]; then
    echo "  [FAIL] 磁盘不够（${AVAIL_GB} < ${DISK_NEED_GB} GB）。三个选择："
    echo "         ① PROF_DIR=<更大分区>/prof_128k  ② PROF_LEVEL=0  ③ PROF_NO_AIC=1"
    exit 1
  fi
fi
echo "  工作目录  : ${OUT}"
echo "  证据包 TAG: ${TAG}"
echo

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[DRY] 会依次执行："
  echo "  1) 起 server: DP_SIZE=$DP_SIZE CHUNKED_PREFILL=$CHUNKED_PREFILL MEM_FRACTION=$MEM_FRACTION DECODE_GRAPH=$DECODE_GRAPH MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-131072} EXTRA_ENV=[SGLANG_NPU_PROFILE_DP_REPLICA=$DP_REPLICA SGLANG_NPU_PROFILE_SIMPLIFY=$SIMPLIFY ${EXTRA_ENV:-}] EXTRA_ARGS=[${EXTRA_ARGS:-}]"
  if [ "$WARMUP" -gt 0 ]; then
    echo "  2) 预热: WARMUP=$WARMUP NUM_PROMPTS=1 CACHE_PCT=$CACHE_PCT OUT_LEN=$OUT_LEN bench_128k_1k.sh"
  else
    echo "  2) 跳过预热（WARMUP=0，冷启动）"
  fi
  echo "  3) POST /start_profile {output_dir:$PROF_DIR, num_steps:$NUM_STEPS, activities:[CPU,GPU], with_stack:false}  + Bearer ${ADMIN_KEY}"
  echo "  4) 采集: WARMUP=0 SKIP_FLUSH=1 NUM_PROMPTS=1 CACHE_PCT=$CACHE_PCT OUT_LEN=$OUT_LEN bench_128k_1k.sh（冷启动⇒命中应≈0%）"
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
  python3 - "$1" "$2" "${3:-120}" "$ADMIN_KEY" <<'PYAPI'
import http.client, sys
path, body, tmo, key = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
hdr = {"Content-Type": "application/json"}
if key:
    hdr["Authorization"] = "Bearer " + key
c = http.client.HTTPConnection("127.0.0.1", 5684, timeout=tmo)
c.request("POST", path, body.encode() or b"{}", hdr)
print(c.getresponse().read().decode().strip()[:300])
PYAPI
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
[ "$DP_REPLICA" = "1" ] && PROF_ENV="SGLANG_NPU_PROFILE_DP_REPLICA=1"
[ "$RANK0_ONLY" = "1" ] && PROF_ENV="${PROF_ENV:+$PROF_ENV }SGLANG_NPU_PROFILE_RANK0_ONLY=1"
[ "$SIMPLIFY" = "1" ]   && PROF_ENV="${PROF_ENV:+$PROF_ENV }SGLANG_NPU_PROFILE_SIMPLIFY=1"
[ -n "$PROF_LEVEL" ]    && PROF_ENV="${PROF_ENV:+$PROF_ENV }SGLANG_NPU_PROFILE_LEVEL=${PROF_LEVEL}"
[ "$PROF_NO_AIC" = "1" ] && PROF_ENV="${PROF_ENV:+$PROF_ENV }SGLANG_NPU_PROFILE_NO_AIC=1"

# ⚠ 调用方传进来的 EXTRA_ENV 必须与 profiling 旋钮**合并**，不能二选一。
#   本脚本此前只把自己的 PROF_ENV 传给启动器，完全丢弃调用方的 EXTRA_ENV ⇒
#   SGLANG_DSPARK_NVLINK_VOCAB_GATHER=0 / HCCL_BUFFSIZE 这类必需 env 静默失效
#   （前者不生效就是在 vocab_gather 里 pynvml NameError 崩，见 HANDOFF §12 坑 #2）。
#   EXTRA_ARGS（CLI 参数）另走一条路，由启动器的 ${EXTRA_ARGS} 消费，见 run_dsv41_w4a8.sh。
FULL_ENV="$PROF_ENV"
if [ -n "${EXTRA_ENV:-}" ]; then
  FULL_ENV="${FULL_ENV:+$FULL_ENV }${EXTRA_ENV}"
fi
echo "    EXTRA_ENV =${FULL_ENV:-（无）}"
[ -n "$EXTRA_ARGS" ] && echo "    EXTRA_ARGS=${EXTRA_ARGS}"
case "$DP_REPLICA$RANK0_ONLY" in
  11) echo "    [warn] DP_REPLICA=1 与 RANK0_ONLY=1 同时开：两门控是「或」关系，等同只采 tp_rank 0" ;;
esac
EXPECT_COLLECT=1
[ "$DP_REPLICA" = "1" ] && EXPECT_COLLECT="$DP_SIZE"
EXPECT_SKIP=$((TP_SIZE - EXPECT_COLLECT))
echo "    期望采集 rank 数=${EXPECT_COLLECT}（跳过 ${EXPECT_SKIP}/${TP_SIZE}）"
stage "launch server dp=$DP_SIZE chunk=$CHUNKED_PREFILL mem=$MEM_FRACTION graph=$DECODE_GRAPH extra=[$FULL_ENV] args=[$EXTRA_ARGS]"

( cd "$SEG" && env DP_SIZE="$DP_SIZE" CHUNKED_PREFILL="$CHUNKED_PREFILL" \
    MEM_FRACTION="$MEM_FRACTION" DECODE_GRAPH="$DECODE_GRAPH" \
    MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-131072}" \
    ${DECODE_BS:+DECODE_BS="$DECODE_BS"} \
    ${FULL_ENV:+EXTRA_ENV="$FULL_ENV"} \
    EXTRA_ARGS="--admin-api-key ${ADMIN_KEY}${EXTRA_ARGS:+ $EXTRA_ARGS}" \
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
grep -E "^  (tp-size|mem-fraction|decode graph|chunked prefill|extra env|extra args|max prefill tok)" "$SLOG" | sed 's/^/      /'

# 早失败检查（约 5 分钟时就判定，别等到 25 分钟后才发现旋钮没生效）
KNOBS_OK=1
for kv in $FULL_ENV; do
  if ! grep -qF -- "$kv" "$SLOG"; then
    echo "  [FAIL] 启动器日志的 'extra env' 里没有 ${kv} —— EXTRA_ENV 没传到 server"
    KNOBS_OK=0
  fi
done
if [ "$KNOBS_OK" != "1" ]; then
  grep -E "^  extra env" "$SLOG" | sed 's/^/      日志实际: /'
  tail -20 "$SLOG"; stage "FAIL env knobs not applied"; cleanup; exit 1
fi
echo "    ✓ EXTRA_ENV 全部 ${#FULL_ENV} 字符已落到 server（逐项 grep 命中）"

# ---------------- 2. 预热（可选，默认跳过） ----------------
echo
if [ "$WARMUP" -gt 0 ]; then
  echo "[2/7] 预热：${WARMUP} 条（CACHE_PCT=${CACHE_PCT}）"
  echo "      ⚠ 注意：本脚本的预热送的是**与采集条相同的 prompt**，它会把整条 prompt 灌进缓存，"
  echo "        采集条随即变成 100% 命中、没有 prefill。要采「50% 命中」必须让预热只灌**共享前缀**"
  echo "        （CACHE_PCT=100 + IN_LEN=SHARED），见 HANDOFF §15.6；默认走冷启动。"
  stage "warmup ${WARMUP} requests"
  if ! run_with_heartbeat "预热" "$OUT/warmup.log" \
      env WARMUP="$WARMUP" NUM_PROMPTS=1 CONCURRENCY=1 DP_SIZE="$DP_SIZE" \
          CACHE_PCT="$CACHE_PCT" OUT_LEN="$OUT_LEN" ${IN_LEN:+IN_LEN="$IN_LEN"} \
          bash "$HERE/bench_128k_1k.sh"; then
    echo "  [FAIL] 预热失败，日志：$OUT/warmup.log"; tail -30 "$OUT/warmup.log"; stage "FAIL warmup"; exit 1
  fi
  grep -hE "^HIT=|^TTFT_" "$OUT/warmup.log" | tail -2 | sed 's/^/    /'
else
  echo "[2/7] 跳过预热（WARMUP=0）：冷启动，采集条的命中率应为 0%，一次跑满 32 个 chunk"
  stage "warmup skipped (cold)"
fi

# ---------------- 3. 开 profile ----------------
echo
echo "[3/7] 开 profile（num_steps=${NUM_STEPS}，activities=[CPU,GPU]，with_stack=false）"
stage "start_profile num_steps=$NUM_STEPS"
SP_RESP=$(api_post "/start_profile" "{\"output_dir\":\"${PROF_DIR}\",\"num_steps\":${NUM_STEPS},\"activities\":[\"CPU\",\"GPU\"],\"with_stack\":false}" 60)
echo "    /start_profile → ${SP_RESP}"
case "$SP_RESP" in
  *error*|*Error*)
    echo "  [FAIL] /start_profile 被拒（多半是 admin-api-key 鉴权）。"
    echo "         /start_profile 是 admin_force：server 要带 --admin-api-key，POST 要带 Bearer。"
    stage "FAIL start_profile rejected"; cleanup; exit 1 ;;
esac
sleep 2
# grep -c 命中 0 次时会「打印 0 + 退出码 1」，所以只能 `|| true`，不能再 `|| echo 0`（会出现 0\n0）
N_START=$(grep -c "Profiling starts" "$SLOG" 2>/dev/null || true); N_START=${N_START:-0}
N_SKIP=$(grep -c "NPU profiling skipped" "$SLOG" 2>/dev/null || true); N_SKIP=${N_SKIP:-0}
echo "    日志里 Profiling starts=${N_START}（期望 ${TP_SIZE}：这行打在门控之前），skipped=${N_SKIP}（期望 ${EXPECT_SKIP}）"
if [ "${N_START:-0}" -eq 0 ]; then
  echo "  [FAIL] /start_profile 没有生效（日志里没有 'Profiling starts'）"; stage "FAIL start_profile"; exit 1
fi
if [ "${N_SKIP:-0}" -ne "$EXPECT_SKIP" ]; then
  echo "  [warn] skipped 数不对：拿到 ${N_SKIP}，期望 ${EXPECT_SKIP}（门控或 attn_tp 推导有问题）"
fi
grep "NPU profiler config" "$SLOG" | head -1 | sed 's/^/    /'

# ---------------- 4. 发采集请求 ----------------
echo
if [ "$WARMUP" -gt 0 ]; then
  echo "[4/7] 发 1 条 128K 请求（SKIP_FLUSH=1：复用预热好的前缀缓存；期望命中 ${CACHE_PCT}%）"
else
  echo "[4/7] 发 1 条 128K **冷**请求（新 server 空缓存；期望命中 0%，32 个 chunk，TTFT≈110s）"
fi
stage "profiled request (1 prompt, out_len=$OUT_LEN)"
if ! run_with_heartbeat "采集" "$OUT/bench.log" \
    env WARMUP=0 SKIP_FLUSH=1 NUM_PROMPTS=1 CONCURRENCY=1 DP_SIZE="$DP_SIZE" \
        CACHE_PCT="$CACHE_PCT" OUT_LEN="$OUT_LEN" ${IN_LEN:+IN_LEN="$IN_LEN"} \
        bash "$HERE/bench_128k_1k.sh"; then
  echo "  [WARN] 采集请求返回非 0（可能是 num_steps 提前触发的正常现象），继续看 stop 结果"
fi
grep -hE "^HIT=|^TTFT_|^OUT_TPUT=" "$OUT/bench.log" | tail -3 | sed 's/^/    /'

# 命中率守卫：冷启动就该是 0% 左右；若接近 100% 说明缓存里已有整条 prompt，
# 这一轮的 trace 里没有 prefill —— 直接判失败，别浪费导出和后续分析。
HIT_VAL=$(grep -hoE '^HIT=[0-9.]+' "$OUT/bench.log" 2>/dev/null | tail -1 | cut -d= -f2)
if [ -z "${HIT_VAL:-}" ]; then
  echo "  [warn] 没从 bench 日志里读到 HIT，跳过命中率守卫"
else
  if [ "$WARMUP" -gt 0 ]; then HIT_LO=35; HIT_HI=65; else HIT_LO=0; HIT_HI=10; fi
  echo "    命中率守卫：实测 ${HIT_VAL}%（本轮期望 ${HIT_LO}~${HIT_HI}%）"
  if awk "BEGIN{exit !(${HIT_VAL} >= ${HIT_LO} && ${HIT_VAL} <= ${HIT_HI})}"; then
    :
  else
    echo "  [FAIL] 命中率 ${HIT_VAL}% 不在 ${HIT_LO}~${HIT_HI}% 内：这一轮的 trace 里 prefill 成分不对"
    echo "         冷启动却高命中 ⇒ 缓存没清干净（注意：设了 --admin-api-key 之后 /flush_cache 也要 Bearer，"
    echo "         bench 的 flush 会 401 而被跳过）。请确认是**新起的 server**，或手动加 header flush。"
    stage "FAIL hit out of band ${HIT_VAL}"; PACKAGE=0
  fi
fi

# ---------------- 5. 停 profile ----------------
echo
echo "[5/7] 停 profile（导出+解析，128K 可能要 1~3 分钟）"
# num_steps 到点后 profiler 会**自动** stop + 导出（scheduler 里按 forward ct 触发）。
# 这时再 POST /stop_profile 只会得到 "Profiling is not in progress"（它在 barrier 之前就返回，
# 不会卡，但日志会很迷惑），所以先看日志再决定要不要发。
if grep -q "Profiling done" "$SLOG" 2>/dev/null; then
  echo "  ✓ num_steps 到点已自动停止并导出（跳过 POST /stop_profile）"
  stage "auto-stopped at num_steps"
else
  stage "stop_profile"
  api_post "/stop_profile" "{}" 900
fi
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
if [ "${PACKAGE:-1}" = "0" ]; then
  echo "  [skip] 命中率不合格（见上），不打包；trace 仍保留在 ${PROF_DIR} 供排查"
  stage "skip packaging"
  echo
  echo "===== 完成（未打包）====="
  echo "trace 目录 : ${PROF_DIR}"
  [ "$KEEP_SERVER" = "1" ] || { cleanup; sleep 2; }
  trap - EXIT INT TERM
  exit 3
fi
if [ ! -d "$PROF_DIR" ]; then
  echo "  [FAIL] trace 目录不存在：${PROF_DIR}"; tail -40 "$SLOG"; stage "FAIL no trace dir"; exit 1
fi
du -sh "$PROF_DIR" | sed 's/^/    trace 目录: /'
echo "    rank 目录（每个 DP replica 一个；只有「有请求的那个 replica」会很大，其余是空闲 rank）："
for d in "$PROF_DIR"/*_ascend_pt; do
  [ -d "$d" ] || continue
  kd="$d/ASCEND_PROFILER_OUTPUT/kernel_details.csv"
  printf '      %-12s %s  kernel_details=%s\n' \
    "$(du -sh "$d" 2>/dev/null | cut -f1)" \
    "$(basename "$d")" \
    "$([ -f "$kd" ] && du -h "$kd" 2>/dev/null | cut -f1 || echo 无)"
done
N_DIRS=$(ls -d "$PROF_DIR"/*_ascend_pt 2>/dev/null | wc -l | tr -d ' ')
if [ "$N_DIRS" -ne "$EXPECT_COLLECT" ]; then
  echo "  [warn] rank 目录数 ${N_DIRS} ≠ 期望 ${EXPECT_COLLECT}（门控没按预期生效？）"
fi

# 先跑一次按 chunk 的归因，把**诊断信息**也存进证据包。
# 3 步的 smoke 不足以拟合斜率，但它能告诉我们切步对不对、桶分布合不合理——
# 这正是 smoke 的价值：拿它把分析器改对，而不是拿 32 步的大 trace 试错。
ANALYZE=${ANALYZE:-1}
if [ "$ANALYZE" = "1" ] && [ -f "$HERE/analyze_prefill_steps.py" ]; then
  python3 "$HERE/analyze_prefill_steps.py" "$PROF_DIR" \
      --chunk-tokens $((CHUNKED_PREFILL / DP_SIZE)) --cached-prefix 0 \
      --window-from $((CHUNKED_PREFILL / DP_SIZE * 17)) 2>&1 | tee "$OUT/prefill_steps.txt" | sed 's/^/    /'
else
  echo "    （跳过 analyze_prefill_steps.py）"
fi

python3 "$HERE/collect_profile_bundle.py" "$PROF_DIR" --tag "$TAG" | sed 's/^/    /'
BUNDLE_DIR="$HERE/profile_bundle/$TAG"
if [ -f "$OUT/prefill_steps.txt" ]; then
  cp "$OUT/prefill_steps.txt" "$BUNDLE_DIR/prefill_steps.txt"
  cp "$OUT/bench.log" "$BUNDLE_DIR/bench.log" 2>/dev/null || true
fi
du -sh "$BUNDLE_DIR" 2>/dev/null | sed 's/^/    证据包: /'

INSIGHT_TGZ="/tmp/${TAG}_insight.tgz"
if [ "$MAKE_TGZ" = "1" ]; then
  # 只压「最有料」的那个 rank 目录：MindStudio Insight 一次只开一个 *_ascend_pt，
  # 把 4 个都压进去是 4 倍体积（3 个还是空闲 replica 的）。
  BEST_RANK=""
  BEST_KB=0
  for d in "$PROF_DIR"/*_ascend_pt; do
    [ -d "$d" ] || continue
    kb=$(du -sk "$d" 2>/dev/null | cut -f1)
    [ "${kb:-0}" -gt "$BEST_KB" ] && { BEST_KB=$kb; BEST_RANK=$d; }
  done
  if [ -n "$BEST_RANK" ]; then
    tar czf "$INSIGHT_TGZ" -C "$(dirname "$BEST_RANK")" "$(basename "$BEST_RANK")" 2>/dev/null
    echo "    Insight 包: ${INSIGHT_TGZ}  ($(du -h "$INSIGHT_TGZ" 2>/dev/null | cut -f1)，只含 $(basename "$BEST_RANK"))"
    echo "    解压后把这个目录交给 Insight：$(basename "$BEST_RANK")"
  else
    echo "    [warn] 找不到 *_ascend_pt 目录，跳过 tgz"
  fi
else
  echo "    Insight 包: 跳过（MAKE_TGZ=0；要看时间线就 MAKE_TGZ=1，它会占额外 2~4GB）"
fi
stage "collected bundle=$(du -sh "$BUNDLE_DIR" 2>/dev/null | cut -f1) tgz=$(du -h "$INSIGHT_TGZ" 2>/dev/null | cut -f1)"

# ---------------- 7. 汇总 ----------------
echo
echo "===== 完成（复制这段回传）====="
echo "TAG=${TAG}  dp=${DP_SIZE} chunk/rank=$((CHUNKED_PREFILL / DP_SIZE)) mem=${MEM_FRACTION} cache=${CACHE_PCT}%"
echo "num_steps=${NUM_STEPS} dp_replica=${DP_REPLICA} rank0_only=${RANK0_ONLY} simplify=${SIMPLIFY}"
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
