#!/usr/bin/env bash
# 过夜扫参：每个配置 = 重启 server + 一次 128K/1k bench，最后出对照表。
#
# 设计要点（无人值守）：
#   * 每个配置都有超时（PER_CONFIG_TIMEOUT），卡住就跳过，不会挂死整晚
#   * 每个配置失败也不中断，记 status=FAIL 继续下一个
#   * 记录每个配置的服务端 facts（池子/chunk/可用显存），便于归因
#   * 脚本退出/被打断时自动 pkill，不会留下 server 占着机器
#   * MAX_HOURS 到点自动收尾，第二天早上一定已经停了
#
# 用法：
#   bash _tmp_preflight/sweep_ttft.sh                    # 跑默认 8 个配置
#   MAX_HOURS=6 bash _tmp_preflight/sweep_ttft.sh
#   DRY_RUN=1 bash _tmp_preflight/sweep_ttft.sh           # 只验证流程/产出表格，不真跑
#   SWEEP_ENVS="DECODE_GRAPH=0" bash _tmp_preflight/sweep_ttft.sh   # 每个配置都追加的 env
#
# 产出（默认 /tmp/dsv41_sweep/）：
#   results.csv        每配置一行
#   report.md          markdown 表格（按 TTFT 升序）
#   <name>.server.log  server 启动日志
#   <name>.bench.log   bench 日志
set -u

OUT_DIR=${OUT_DIR:-/tmp/dsv41_sweep}
MAX_HOURS=${MAX_HOURS:-8}
PER_CONFIG_TIMEOUT=${PER_CONFIG_TIMEOUT:-2400}   # 40 分钟/配置
NUM_PROMPTS=${NUM_PROMPTS:-8}
CONCURRENCY=${CONCURRENCY:-1}
DRY_RUN=${DRY_RUN:-0}
HERE=$(cd "$(dirname "$0")" && pwd)
START_TS=$(date +%s)

mkdir -p "$OUT_DIR"
CSV="$OUT_DIR/results.csv"
REPORT="$OUT_DIR/report.md"
echo "name,dp_size,chunk_global,chunk_per_rank,mem_fraction,spec,prefill_graph,cache_pct,max_total_num_tokens,avail_gb,hit,ttft_ms,tpot_ms,tput,complete,status" > "$CSV"

cleanup() {
  if [ "$DRY_RUN" != "1" ]; then
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    pkill -9 python 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

elapsed_h() { echo $(( ($(date +%s) - START_TS) / 3600 )); }

# name|dp|chunk_global|mem|spec(1=on)|prefill_graph(1=on)
CONFIGS=${CONFIGS:-'
baseline|8|32768|0.85|1|0
chunk8192|8|65536|0.85|1|0
chunk16384|8|131072|0.85|1|0
mem088|8|32768|0.88|1|0
mem090|8|32768|0.90|1|0
dp4_chunk4096|4|16384|0.85|1|0
dp2_chunk4096|2|8192|0.85|1|0
prefill_graph|8|32768|0.85|1|1
'}

run_one() {
  local name=$1 dp=$2 chunk=$3 mem=$4 spec=$5 pgraph=$6 cachepct=${7:-50}
  local slog="$OUT_DIR/$name.server.log"
  local blog="$OUT_DIR/$name.bench.log"

  echo
  echo "############ [$name] dp=$dp chunk=$chunk mem=$mem spec=$spec pgraph=$pgraph ############"

  if [ "$DRY_RUN" = "1" ]; then
    # 假数据，只验证流程
    echo "FAKE max_total_num_tokens=2000000, chunked_prefill_size=$((chunk/dp)), available_gpu_mem=5.00 GB" > "$slog"
    printf 'HIT=50.0%% TARGET=50%%\nTTFT_mean=%d ms (SLO 30000)\nTPOT_mean=33 ms (SLO 50)\nOUT_TPUT=10 tok/s  CONC=1  COMPLETE=True\n' $((60000 + chunk / 100)) > "$blog"
  else
    cleanup
    sleep 3
    local envs="DP_SIZE=$dp CHUNKED_PREFILL=$chunk MEM_FRACTION=$mem"
    [ "$spec" = "1" ] || envs="$envs DISABLE_SPEC=1"
    [ "$pgraph" = "1" ] && envs="$envs PREFILL_GRAPH=1"
    [ -n "${EXTRA_ENV_CFG:-}" ] && envs="$envs EXTRA_ENV=\"$EXTRA_ENV_CFG\""
    [ -n "${SWEEP_ENVS:-}" ] && envs="$envs $SWEEP_ENVS"
    # shellcheck disable=SC2086
    ( env $envs timeout "$PER_CONFIG_TIMEOUT" bash "$HERE/run_dsv41_w4a8.sh" ) > "$slog" 2>&1 &
    local srv_pid=$!

    # 等 ready（最多 20 分钟；加载 w4a8 约 2-4 分钟，JIT 缓存热时更快）
    local waited=0 ok=0
    while [ "$waited" -lt 1200 ]; do
      if curl --noproxy '*' -fsS -m 5 "http://127.0.0.1:5684/v1/models" >/dev/null 2>&1; then ok=1; break; fi
      if ! kill -0 "$srv_pid" 2>/dev/null; then break; fi
      sleep 10; waited=$((waited + 10))
    done
    if [ "$ok" != "1" ]; then
      echo "  [FAIL] server 未就绪（等待 ${waited}s），跳过"
      echo "$name,$dp,$chunk,$((chunk/dp)),$mem,$spec,$pgraph,$cachepct,,,,,,,FAIL_NOT_READY" >> "$CSV"
      cleanup
      return
    fi
    echo "  server ready（等待 ${waited}s）"

    # 等 warmup 覆盖所有 DP rank（前缀缓存按 rank 分区）
    local W=$dp
    # shellcheck disable=SC2086
    WARMUP=$W NUM_PROMPTS=$NUM_PROMPTS CONCURRENCY=$CONCURRENCY DP_SIZE=$dp CACHE_PCT=$cachepct \
      timeout "$PER_CONFIG_TIMEOUT" bash "$HERE/bench_128k_1k.sh" > "$blog" 2>&1
    echo "  bench rc=$?"
  fi

  local pool chunk_eff avail hit ttft tpot tput comp status
  pool=$(grep -oE "max_total_num_tokens=[0-9]+" "$slog" | head -1 | cut -d= -f2)
  chunk_eff=$(grep -oE "chunked_prefill_size=[0-9]+" "$slog" | head -1 | cut -d= -f2)
  avail=$(grep -oE "available_gpu_mem=[0-9.]+" "$slog" | head -1 | cut -d= -f2)
  hit=$(grep -o 'HIT=[^ ]*' "$blog" | tail -1 | cut -d= -f2)
  ttft=$(grep -o 'TTFT_mean=[0-9.]*' "$blog" | tail -1 | cut -d= -f2)
  tpot=$(grep -o 'TPOT_mean=[0-9.]*' "$blog" | tail -1 | cut -d= -f2)
  tput=$(grep -o 'OUT_TPUT=[0-9.]*' "$blog" | tail -1 | cut -d= -f2)
  comp=$(grep -o 'COMPLETE=[A-Za-z]*' "$blog" | tail -1 | cut -d= -f2)
  status=OK
  [ -n "${ttft:-}" ] || status=FAIL_NO_RESULT
  [ "${comp:-}" = "True" ] || status=INCOMPLETE

  echo "  pool=${pool:-?} chunk=${chunk_eff:-?} avail=${avail:-?}GB HIT=${hit:-?} TTFT=${ttft:-?} TPOT=${tpot:-?} ${status}"
  echo "$name,$dp,$chunk,$((chunk/dp)),$mem,$spec,$pgraph,$cachepct,${pool:-},${avail:-},${hit:-},${ttft:-},${tpot:-},${tput:-},${comp:-},$status" >> "$CSV"

  cleanup
  sleep 5
}

echo "===== 扫参开始 $(date) | 输出 $OUT_DIR | MAX_HOURS=$MAX_HOURS ====="
for cfg in $CONFIGS; do
  [ -z "$cfg" ] && continue
  if [ "$(elapsed_h)" -ge "$MAX_HOURS" ]; then
    echo "[stop] 已跑 $(elapsed_h) 小时，达到 MAX_HOURS=${MAX_HOURS}，收尾"
    break
  fi
  IFS='|' read -r n d c m s p cp <<< "$cfg"
  run_one "$n" "$d" "$c" "$m" "$s" "$p" "${cp:-50}"
done

# ---- 报告 ----
{
  echo "# 128K/1k TTFT 扫参报告"
  echo
  echo "- 生成时间：$(date)"
  echo "- 条件：输入 130,048（50% 共享前缀）、输出 1,024、并发 ${CONCURRENCY}、每配置 $NUM_PROMPTS 条"
  echo "- SLO：TTFT < 30000 ms，TPOT < 50 ms"
  echo
  echo "| 配置 | dp | chunk(全局/每rank) | mem | spec | prefill图 | cache% | 池子 | avail | HIT | TTFT(ms) | TPOT(ms) | 状态 |"
  echo "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
  tail -n +2 "$CSV" | while IFS=, read -r name dp cg cpr mem spec pg cp pool avail hit ttft tpot tput comp status; do
    printf "| %s | %s | %s / %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" \
      "$name" "$dp" "$cg" "$cpr" "$mem" "$spec" "$pg" "$cp" "${pool:--}" "${avail:--}" "${hit:--}" "${ttft:--}" "${tpot:--}" "$status"
  done
  echo
  echo "明细日志：$OUT_DIR"
} > "$REPORT"

echo
echo "===== 完成 $(date) ====="
cat "$REPORT"
