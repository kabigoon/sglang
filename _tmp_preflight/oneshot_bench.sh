#!/usr/bin/env bash
# 一次性实验：起 server → 等 ready → 跑 bench → 打回传块 → （默认）停 server。
# 机器不独占时用这个：一次一条命令，跑完就释放。
#
# 用法（一次尝试）：
#   DP_SIZE=2 bash _tmp_preflight/oneshot_bench.sh                    # 每 rank 4096 自动推导
#   DP_SIZE=4 bash _tmp_preflight/oneshot_bench.sh
#   DP_SIZE=1 CHUNKED_PREFILL=4096 bash _tmp_preflight/oneshot_bench.sh
#   KEEP_SERVER=1 DP_SIZE=2 bash _tmp_preflight/oneshot_bench.sh       # 跑完保留 server
#
# 标度曲线（同一个 server，只变上下文长度，省掉每次 3-5 分钟的起服务）：
#   DP_SIZE=4 CHUNKED_PREFILL=16384 DECODE_GRAPH=1 DECODE_BS="1 2 4 8 16" \
#     WARMUP=4 NUM_PROMPTS=8 CONCURRENCY=1 CACHE_PCT=50 SCALE="65536 98304 126250" \
#     bash _tmp_preflight/oneshot_bench.sh
#
# 单点指定输入长度（不透传 IN_LEN 时由 bench 脚本推导）：
#   IN_LEN=98304 bash _tmp_preflight/oneshot_bench.sh
#
# 算子开关扫描（每轮只加一个 K=V，且必须过 GSM8K 才认）：
#   EXTRA_ENV="SGLANG_OPT_FUSE_WQA_WKV=1" bash _tmp_preflight/oneshot_bench.sh
#
# 已知：dp != 8 时 decode 图会 AssertionError: capture_bs=[]，本脚本自动 DECODE_GRAPH=0；
#       想保住 decode 图（TPOT 才与 §5.2 可比）显式给 DECODE_GRAPH=1 DECODE_BS="1 2 4 8 16"。
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SEG=$(cd "$HERE/.." && pwd)

DP_SIZE=${DP_SIZE:-8}
CHUNKED_PREFILL=${CHUNKED_PREFILL:-$((4096 * DP_SIZE))}   # 保持每 DP rank 4096
MEM_FRACTION=${MEM_FRACTION:-0.85}
if [ -z "${DECODE_GRAPH:-}" ]; then
  if [ "$DP_SIZE" -eq 8 ]; then DECODE_GRAPH=1; else DECODE_GRAPH=0; fi
fi
DECODE_BS=${DECODE_BS:-}
EXTRA_ENV=${EXTRA_ENV:-}
DISABLE_SPEC=${DISABLE_SPEC:-0}
NUM_PROMPTS=${NUM_PROMPTS:-8}
WARMUP=${WARMUP:-$DP_SIZE}          # 覆盖所有 DP rank 的前缀缓存
CACHE_PCT=${CACHE_PCT:-50}          # 命中率固定 50%，不改
CONCURRENCY=${CONCURRENCY:-1}
IN_LEN=${IN_LEN:-}                  # 空=让 bench 脚本自己推导（默认 128K 自适应）
OUT_LEN=${OUT_LEN:-}
SKIP_FLUSH=${SKIP_FLUSH:-}
SCALE=${SCALE:-}                    # 非空=同一 server 依次跑这些名义输入长度
HB=${HB:-30}                        # bench 阶段心跳间隔秒（测试时可调小）
READY_TIMEOUT=${READY_TIMEOUT:-900}   # 等 ready 的上限秒
PROBE_HB=${PROBE_HB:-30}              # 等 ready 阶段每多少秒打印一次探测结果
KEEP_SERVER=${KEEP_SERVER:-0}
DRY_RUN=${DRY_RUN:-0}

OUT=${OUT:-/tmp/oneshot_$(date +%m%d_%H%M%S)_dp${DP_SIZE}}
mkdir -p "$OUT"
SLOG="$OUT/server.log"
# 轨迹文件：脚本万一被外部杀掉（会话断、SIGKILL，trap 不执行），
# cat 这个文件就知道它跑到哪一步、几点停的。追加写，保留全过程。
STAGE="$OUT/stage.txt"
stage() { printf '%s | %s\n' "$(date '+%F %T')" "$*" >> "$STAGE" 2>/dev/null || true; }

echo "===== 本次尝试 ====="
# 版本号：机器上一眼看出跑的是不是最新脚本（没 pull 会显示旧 sha）
echo "  脚本版本: $(git -C "$SEG" rev-parse --short HEAD 2>/dev/null || echo 未知)  ($HERE)"
echo "  工具: python3=$(command -v python3 >/dev/null && echo ok || echo 缺) curl=$(command -v curl >/dev/null && echo ok || echo 缺) pkill=$(command -v pkill >/dev/null && echo ok || echo 缺) pgrep=$(command -v pgrep >/dev/null && echo ok || echo 缺)"
echo "  dp=$DP_SIZE  chunk=${CHUNKED_PREFILL}(全局)/$((CHUNKED_PREFILL / DP_SIZE))(每rank)  mem=$MEM_FRACTION"
echo "  decode_graph=$DECODE_GRAPH  bs=[${DECODE_BS:-默认}]  spec=$([ "$DISABLE_SPEC" = "1" ] && echo off || echo on)"
echo "  CACHE_PCT=$CACHE_PCT  warmup=$WARMUP  prompts=$NUM_PROMPTS  conc=$CONCURRENCY"
[ -n "$SCALE" ] && echo "  标度曲线 SCALE=[$SCALE]（同一个 server，只变上下文长度）"
[ -n "$IN_LEN" ] && echo "  IN_LEN=$IN_LEN"
[ -n "$EXTRA_ENV" ] && echo "  EXTRA_ENV=$EXTRA_ENV   ← 本轮唯一变量"
echo "  输出目录: $OUT"
echo "  轨迹文件: ${STAGE}   ← 万一脚本被杀，看这个就知道死在哪一步"
stage "start pid=$$ dp=$DP_SIZE scale=[${SCALE:-}] in_len=[${IN_LEN:-}] cache=${CACHE_PCT}%"
echo

if [ "$DRY_RUN" = "1" ]; then
  echo "[DRY] 会执行："
  echo "  env DP_SIZE=$DP_SIZE CHUNKED_PREFILL=$CHUNKED_PREFILL MEM_FRACTION=$MEM_FRACTION DECODE_GRAPH=$DECODE_GRAPH bash $HERE/run_dsv41_w4a8.sh"
  echo "  env WARMUP=$WARMUP NUM_PROMPTS=$NUM_PROMPTS CONCURRENCY=$CONCURRENCY DP_SIZE=$DP_SIZE CACHE_PCT=$CACHE_PCT IN_LEN=${IN_LEN:-<auto>} bash $HERE/bench_128k_1k.sh"
  if [ -n "$SCALE" ]; then
    for L in $SCALE; do
      echo "  env WARMUP=$WARMUP NUM_PROMPTS=$NUM_PROMPTS CONCURRENCY=$CONCURRENCY DP_SIZE=$DP_SIZE CACHE_PCT=$CACHE_PCT IN_LEN=$L bash $HERE/bench_128k_1k.sh"
    done
  fi
  exit 0
fi

cleanup() {
  # 先杀掉启动器子 shell：否则我们退出后它仍会继续跑下去、把 server 拉起来变成孤儿
  # （2026-11-23 那次 134 分钟孤儿 server 就是这么产生的）
  if [ -n "${LAUNCH_PID:-}" ]; then
    if command -v pkill >/dev/null 2>&1; then
      pkill -9 -P "$LAUNCH_PID" 2>/dev/null || true
    fi
    kill -9 "$LAUNCH_PID" 2>/dev/null || true
  fi
  if command -v pkill >/dev/null 2>&1; then
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
  elif command -v pgrep >/dev/null 2>&1; then
    for _p in $(pgrep -f "sglang.launch_server" 2>/dev/null); do kill -9 "$_p" 2>/dev/null || true; done
  else
    echo "    [warn] 既没有 pkill 也没有 pgrep，旧 server 得手动 kill"
  fi
}
trap cleanup EXIT INT TERM

# 不依赖 curl 的就绪探测（容器里可能没有 curl；curl 失败时旧版是彻底静默的）。
# 只打印 HTTP 状态码或异常名，成功判据是 200。
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

echo "[1/3] 停旧 server"
stage "stop old server"
cleanup; sleep 3

echo "[2/3] 起 server（日志 ${SLOG}）"
stage "launch server dp=$DP_SIZE chunk=$CHUNKED_PREFILL mem=$MEM_FRACTION graph=$DECODE_GRAPH extra=[${EXTRA_ENV:-}]"

# 注意：EXTRA_ENV / DECODE_BS 的值里含空格（"1 2 4 8 16"、"A=1 B=2"），
# 不能写成 ${VAR:+K="$VAR"}（外层没引号会被 word splitting 拆成多个 env 参数），必须走数组。
SRV_ENV=()
[ -n "$DECODE_BS" ]   && SRV_ENV+=("DECODE_BS=$DECODE_BS")
[ -n "$EXTRA_ENV" ]   && SRV_ENV+=("EXTRA_ENV=$EXTRA_ENV")
[ -n "$DISABLE_SPEC" ] && [ "$DISABLE_SPEC" != "0" ] && SRV_ENV+=("DISABLE_SPEC=$DISABLE_SPEC")
echo "    启动 env: DP_SIZE=$DP_SIZE CHUNKED_PREFILL=$CHUNKED_PREFILL MEM_FRACTION=$MEM_FRACTION DECODE_GRAPH=$DECODE_GRAPH ${SRV_ENV[*]:-}"
( cd "$SEG" && env DP_SIZE="$DP_SIZE" CHUNKED_PREFILL="$CHUNKED_PREFILL" \
    MEM_FRACTION="$MEM_FRACTION" DECODE_GRAPH="$DECODE_GRAPH" \
    ${SRV_ENV[@]+"${SRV_ENV[@]}"} \
    bash "$HERE/run_dsv41_w4a8.sh" ) > "$SLOG" 2>&1 &
LAUNCH_PID=$!
echo "    启动器 pid=${LAUNCH_PID}（前导阶段要 pkill/source/sysctl，几秒到几十秒后才出现 python 进程，属正常）"

RES=""; wait_s=0; last_rep=0
until RES=$(ready_probe) && [ "$RES" = "200" ]; do
  # ⚠ 不能用「pgrep 看不到 sglang.launch_server」当失败判据：启动器在前导阶段
  #   （pkill -9 python、source 4 份 CANN env、sysctl）里还没有 python 进程，
  #   那样会在 0s 就误判失败并退出——然后启动器继续跑，把 server 变成孤儿。
  #   正确判据：启动器子 shell 自己死了，且确实看不到 server 进程。
  launch_alive=1; kill -0 "$LAUNCH_PID" 2>/dev/null || launch_alive=0
  if [ "$launch_alive" = "0" ] && ! pgrep -f "sglang.launch_server" >/dev/null 2>&1; then
    echo "  [FAIL] 启动器已退出且没有 server 进程（等待 ${wait_s}s，最后一次探测=${RES}），日志尾部："
    tail -30 "$SLOG"; stage "FAIL launcher exited after ${wait_s}s probe=$RES"; exit 1
  fi
  sleep 5; wait_s=$((wait_s + 5))
  # 用「距上次打印」而不是取模：sleep 步长 5s，取模在 PROBE_HB 非 5 的倍数时永远不命中
  if [ $((wait_s - last_rep)) -ge "$PROBE_HB" ]; then
    echo "    [${wait_s}s] 等 ready…探测=${RES}  启动器=$([ "$launch_alive" = "1" ] && echo 活 || echo 已退出)  日志末行: $(tail -1 "$SLOG" 2>/dev/null | tr -d '\r' | cut -c1-80)"
    last_rep=$wait_s
  fi
  if [ "$wait_s" -ge "$READY_TIMEOUT" ]; then
    echo "  [FAIL] ${READY_TIMEOUT}s 仍未 ready（探测=${RES}），日志尾部："
    tail -20 "$SLOG"; stage "FAIL ready timeout probe=$RES"; exit 1
  fi
done
echo "  ready（等待 ${wait_s}s，探测=${RES}）"
stage "ready after ${wait_s}s"
grep -oE "max_total_num_tokens=[0-9]+|chunked_prefill_size=[0-9]+|available_gpu_mem=[0-9.]+ GB" "$SLOG" | head -3 | sed 's/^/    /'
echo "    chunked prefill: ${CHUNKED_PREFILL}(全局) -> $((CHUNKED_PREFILL / DP_SIZE)) 每 DP rank"

if [ "$DECODE_GRAPH" = "0" ] && [ "$DP_SIZE" -ne 8 ]; then
  echo "    [提醒] decode 图已关 → TPOT 与 §5.2（graph on，32.8ms）不可比；"
  echo "           要对齐请加 DECODE_GRAPH=1 DECODE_BS=\"1 2 4 8 16\""
fi

# 本轮要跑的输入长度序列：SCALE 优先，其次 IN_LEN，最后交给 bench 自己推导
if [ -n "$SCALE" ]; then
  POINTS="$SCALE"
  echo
  echo "[3/3] 标度曲线：依次跑 IN_LEN = ${POINTS}（server 配置不变，唯一变量=上下文长度）"
else
  POINTS="${IN_LEN:--}"
fi

TSV="$OUT/summary.tsv"
: > "$TSV"
RC_ALL=0
for L in $POINTS; do
  if [ "$L" = "-" ]; then TAG=auto; BENCH_IN_LEN=""; else TAG="$L"; BENCH_IN_LEN="$L"; fi
  BLOG="$OUT/bench_in${TAG}.log"
  echo
  echo "----- 点 IN_LEN=$TAG -----"
  stage "bench IN_LEN=$TAG start"
  echo "  bench 日志: $BLOG"
  echo "  预计 10-15 分钟（128K 点约 15 分钟）。每 ${HB}s 一个心跳，别急着 Ctrl-C。"
  # 不能用 `| tee | tail -30`：tail 要等 EOF 才吐字，屏幕上会静默十几分钟。
  # 改成后台跑 + 父进程心跳，跑完再打末尾 30 行。
  env WARMUP="$WARMUP" NUM_PROMPTS="$NUM_PROMPTS" CONCURRENCY="$CONCURRENCY" \
      DP_SIZE="$DP_SIZE" CACHE_PCT="$CACHE_PCT" \
      ${BENCH_IN_LEN:+IN_LEN="$BENCH_IN_LEN"} \
      ${OUT_LEN:+OUT_LEN="$OUT_LEN"} \
      ${SKIP_FLUSH:+SKIP_FLUSH="$SKIP_FLUSH"} \
      bash "$HERE/bench_128k_1k.sh" > "$BLOG" 2>&1 &
  BPID=$!
  EL=0
  while kill -0 "$BPID" 2>/dev/null; do
    sleep "$HB"
    EL=$((EL + HB))
    LAST=$(grep -v '^[[:space:]]*$' "$BLOG" 2>/dev/null | tail -1 | tr -d '\r' | cut -c1-100)
    echo "    [${EL}s] ${LAST:-（还没有输出）}"
  done
  wait "$BPID"; RC=$?
  echo "  bench 结束：${EL}s，退出码 ${RC}。末尾 30 行："
  tail -30 "$BLOG"
  stage "bench IN_LEN=$TAG end rc=$RC elapsed=${EL}s"
  [ "$RC" -ne 0 ] && RC_ALL=$RC
  # 抽数进汇总（读 bench 落盘的 jsonl，比 grep 文本稳）
  python3 - "$BLOG" "$TAG" "$TSV" <<'PY'
import json, re, sys

log, tag, tsv = sys.argv[1], sys.argv[2], sys.argv[3]
row = [tag, "0", "-", "-", "-", "-", "-", "-", "-", "-"]
try:
    txt = open(log, errors="replace").read()
    paths = re.findall(r"结果文件: (\S+\.jsonl)", txt)
    rows = [l for l in open(paths[-1]) if l.strip()]
    r = json.loads(rows[-1])
    cr = r.get("cache_report") or {}
    n = max(len(r.get("input_lens") or [0]), 1)
    prompt = cr.get("total_prompt_tokens") or 0
    cached = cr.get("total_cached_tokens") or 0
    actual_in = sum(r.get("input_lens") or [0]) / n
    unc = (prompt - cached) / n if prompt else float("nan")
    ttft_mean = r.get("ttft_mean_ms", float("nan")) / 1000.0
    ttft_med = r.get("ttft_median_ms", float("nan")) / 1000.0
    tpot = r.get("tpot_mean_ms", float("nan"))
    tput = r.get("output_throughput", float("nan"))
    hit = cr.get("cache_hit_rate_pct", float("nan"))
    rate = unc / ttft_med if ttft_med else float("nan")
    row = [tag, str(n), "%.0f" % actual_in, "%.2f" % hit, "%.0f" % unc,
           "%.1f" % ttft_mean, "%.1f" % ttft_med, "%.1f" % tpot,
           "%.1f" % tput, "%.1f" % rate]
except Exception as e:  # 失败也占一行，别让表格错位
    row[9] = "ERR"
    sys.stderr.write("[warn] 抽数失败 %s: %s\n" % (tag, e))
open(tsv, "a").write("\t".join(row) + "\n")
PY
done

echo
echo "===== 结果（复制这段） ====="
grep -hE "^HIT=|^TTFT_|^TPOT_|^OUT_TPUT=" "$OUT"/bench_in*.log 2>/dev/null | tail -8
echo
python3 - "$TSV" "$DP_SIZE" "$((CHUNKED_PREFILL / DP_SIZE))" "$MEM_FRACTION" "$DECODE_GRAPH" "$CACHE_PCT" "$SCALE" <<'PY'
import math, sys, time

tsv, dp, chunk, mem, dg, pct, scale = sys.argv[1:8]
rows = [(l.rstrip("\n").split("\t") + ["-"] * 10)[:10] for l in open(tsv)]
title = "标度曲线" if scale else "单点结果"
print("===== %s（复制整段回传） =====" % title)
print("dp=%s chunk/rank=%s mem=%s decode_graph=%s spec=on cache=%s%%  %s"
      % (dp, chunk, mem, dg, pct, time.strftime("%F")))
print("%-9s %5s %10s %6s %8s %11s %10s %7s %9s %8s"
      % ("IN_LEN", "reqs", "actual_in", "HIT%", "U_unc", "TTFT_mean", "TTFT_med", "TPOT", "TPUT", "U/s"))
for c in rows:
    print("%-9s %5s %10s %6s %8s %11s %10s %7s %9s %8s"
          % (c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]))

def num(x):
    return float(x)

good = []
for c in rows:
    try:
        if c[2] != "-" and c[6] != "-" and float(c[2]) > 0 and float(c[6]) > 0:
            good.append(c)
    except ValueError:
        pass

if len(good) >= 2:
    print()
    print("标度指数（用 actual_in 与实际 TTFT_med，检验平滑幂律 vs 路径突变）：")
    for a, b in zip(good, good[1:]):
        T1, T2, t1, t2 = num(a[2]), num(b[2]), num(a[6]), num(b[6])
        print("  %s -> %s : T x%.2f  TTFT x%.2f  =>  指数 %.2f"
              % (a[0], b[0], T2 / T1, t2 / t1, math.log(t2 / t1) / math.log(T2 / T1)))
    print()
    print("判据：指数随长度平滑变化 → 平滑幂律，问题在 kernel 常数因子；")
    print("      某点突跳（该段指数明显高于邻段）→ 那里有路径突变，先查分支/kernel 选择。")

if not scale:
    c = rows[-1]
    try:
        label = "%.0fk/1k" % (float(c[2]) / 1000.0)
    except ValueError:
        label = "?k/1k"
    print()
    print("登记行：")
    print("  %s | cache %s%% | dp=%s chunk/rank=%s mem=%s graph=%s spec=on | HIT=%s%% U=%s "
          "TTFT_mean=%ss TTFT_median=%ss TPOT=%sms TPUT=%s tok/s | %s"
          % (label, pct, dp, chunk, mem, dg, c[3], c[4], c[5], c[6], c[7], c[8],
             time.strftime("%F")))
print("===== 回传块结束 =====")
PY
echo
echo "server log: $SLOG"
echo "汇总 tsv : $TSV"
echo "各点日志 : ${OUT}/bench_in*.log"
echo "（退出码 ${RC_ALL}：0=全部完整性通过）"

stage "done rc=$RC_ALL"
if [ "$KEEP_SERVER" = "1" ]; then
  echo
  echo "KEEP_SERVER=1：server 保留运行（pid $(pgrep -f sglang.launch_server | head -1)）"
  echo "  只跑 client：WARMUP=$WARMUP NUM_PROMPTS=$NUM_PROMPTS CONCURRENCY=$CONCURRENCY DP_SIZE=$DP_SIZE CACHE_PCT=$CACHE_PCT IN_LEN=<长度> bash $HERE/bench_128k_1k.sh"
  trap - EXIT INT TERM
else
  echo
  echo "释放机器：停 server"
fi
