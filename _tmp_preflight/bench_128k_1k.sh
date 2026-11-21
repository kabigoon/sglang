#!/usr/bin/env bash
# 128k/1k, 50% cache, TTFT<30s, TPOT<50ms —— 基线压测 + SLO 判定
#
# 口径（与 32k/1k 90% cache 那套一致）：
#   输入 = --gsp-system-prompt-len（共享前缀） + --gsp-question-len（独有部分）
#   命中率 = sum(cached_tokens)/sum(prompt_tokens)，由 --cache-report 实测
#   TTFT   = 排队 + prefill + 第一步 decode
#   TPOT   = (E2E - TTFT)/(output_len-1)
#
# 用法：
#   bash _tmp_preflight/bench_128k_1k.sh                    # 默认 127k 输入(自动避让上下文上限) + 8 并发
#   CONCURRENCY=4 NUM_PROMPTS=16 bash _tmp_preflight/bench_128k_1k.sh
#   IN_LEN=131072 OUT_LEN=1024 AUTO_FIT=0 bash _tmp_preflight/bench_128k_1k.sh   # 强行 128k 输入
#
# 注意：--warmup-requests 会先用第一条 prompt 打一次（共享前缀因此入 cache），
#      所以实测命中率应接近「名义 50%」；若明显偏低，按末尾提示调 SHARED 再跑。
set -uo pipefail

# ---------------- 可覆盖参数 ----------------
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-5684}
MODEL=${MODEL:-/home/weights/DeepSeek-V4.1-Flash-1017-w8a8}

IN_LEN=${IN_LEN:-131072}          # 名义输入长度 128k
OUT_LEN=${OUT_LEN:-1024}          # 输出长度 1k
CACHE_PCT=${CACHE_PCT:-50}        # 目标前缀命中率(%)
MAX_SEQ=${MAX_SEQ:-131072}        # 服务端 max seq（launch 里 SGLANG_DSV4_GRAPH_MAX_SEQ）
AUTO_FIT=${AUTO_FIT:-1}           # 1=输入+输出超过 MAX_SEQ 时自动收缩输入

NUM_PROMPTS=${NUM_PROMPTS:-32}    # 总请求数
CONCURRENCY=${CONCURRENCY:-8}     # 并发
WARMUP=${WARMUP:-4}               # 预热请求数（让共享前缀入 cache）
SEED=${SEED:-42}

SLO_TTFT_MS=${SLO_TTFT_MS:-30000}
SLO_TPOT_MS=${SLO_TPOT_MS:-50}
SLO_STAT=${SLO_STAT:-mean}        # mean|median|p90|p95|p99
STRICT=${STRICT:-0}               # 1=SLO 不达标则退出码 3

OUT_DIR=${OUT_DIR:-/home/dsv4_test/bench_out}

# ---------------- 推导长度 ----------------
if [ "$AUTO_FIT" = "1" ] && [ $((IN_LEN + OUT_LEN)) -gt "$MAX_SEQ" ]; then
  NEW_IN=$((MAX_SEQ - OUT_LEN))
  echo "[warn] IN_LEN($IN_LEN)+OUT_LEN($OUT_LEN) > MAX_SEQ($MAX_SEQ)，自动改为 IN_LEN=$NEW_IN"
  IN_LEN=$NEW_IN
fi
SHARED=$((IN_LEN * CACHE_PCT / 100))
QUESTION=$((IN_LEN - SHARED))

mkdir -p "$OUT_DIR"
STEM="$OUT_DIR/v41_128k_1k_cache${CACHE_PCT}_c${CONCURRENCY}_n${NUM_PROMPTS}_$(date +%Y%m%d_%H%M%S)"

cat <<EOF
===== [0] 本次压测口径 =====
  endpoint      : http://$HOST:$PORT
  model         : $MODEL
  input         : $IN_LEN  = shared $SHARED ($CACHE_PCT%) + question $QUESTION
  output        : $OUT_LEN
  prompts / 并发: $NUM_PROMPTS / $CONCURRENCY   (warmup=$WARMUP)
  SLO           : TTFT < ${SLO_TTFT_MS}ms, TPOT < ${SLO_TPOT_MS}ms  (按 $SLO_STAT 判定)
  结果          : ${STEM}.jsonl
  日志          : ${STEM}.log
EOF

# ---------------- 前置 ----------------
echo
echo "===== [1] 清缓存 + 健康检查 ====="
curl --noproxy '*' -fsS --connect-timeout 10 --max-time 30 "http://$HOST:$PORT/v1/models" >/dev/null \
  || { echo "服务不可达：http://$HOST:$PORT/v1/models"; exit 1; }
echo "  /v1/models OK"
curl --noproxy '*' -fsS --connect-timeout 10 --max-time 120 -X POST "http://$HOST:$PORT/flush_cache?timeout=60" \
  && echo "  flush_cache OK" || echo "  [!] flush_cache 失败（继续；但命中率可能被上一次残留污染）"

# ---------------- 压测 ----------------
echo
echo "===== [2] 开始压测（前台，原始输出随 tee 打印）====="
python3 -m sglang.benchmark.serving \
  --backend sglang \
  --host "$HOST" --port "$PORT" \
  --model "$MODEL" \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 1 \
  --gsp-prompts-per-group "$NUM_PROMPTS" \
  --gsp-system-prompt-len "$SHARED" \
  --gsp-question-len "$QUESTION" \
  --gsp-output-len "$OUT_LEN" \
  --num-prompts "$NUM_PROMPTS" \
  --max-concurrency "$CONCURRENCY" \
  --request-rate inf \
  --seed "$SEED" \
  --warmup-requests "$WARMUP" \
  --cache-report \
  --output-details \
  --output-file "${STEM}.jsonl" \
  2>&1 | tee "${STEM}.log"

# ---------------- 解析 + SLO 判定 ----------------
echo
echo "===== [3] 结果与 SLO 判定 ====="
python3 - "$STEM.jsonl" "$IN_LEN" "$OUT_LEN" "$CACHE_PCT" "$SLO_TTFT_MS" "$SLO_TPOT_MS" "$SLO_STAT" "$STRICT" <<'PY'
import json, sys

(path, in_len, out_len, cache_pct, slo_ttft, slo_tpot, stat, strict) = sys.argv[1:9]
in_len, out_len, cache_pct = int(in_len), int(out_len), float(cache_pct)
slo_ttft, slo_tpot, strict = float(slo_ttft), float(slo_tpot), strict == "1"

rows = [json.loads(l) for l in open(path) if l.strip()]
r = rows[-1]

def g(name, default=None):
    return r.get(name, default)

print("请求数            : %d" % len(g("output_lens") or []))
print("输入长度 (mean/max): %.0f / %d" % (sum(g("input_lens") or [0]) / max(len(g("input_lens") or [1]), 1), max(g("input_lens") or [0])))
print("输出长度 分布      : %s" % (sorted(set(g("output_lens") or []))[:8],))

# 缓存
cr = g("cache_report") or {}
hit = cr.get("cache_hit_rate_pct")
print("前缀命中率         : %s%%  (目标 %g%%)" % (hit, cache_pct))
if cr:
    print("   cached/prompt    : %s / %s" % (cr.get("total_cached_tokens"), cr.get("total_prompt_tokens")))

# 关键指标
print("\n%-10s %10s %10s %10s %10s %10s" % ("指标", "mean", "median", "p90", "p99", "max"))
def line(label, prefix, lst=None):
    vals = [r.get("%s_%s_ms" % (k, prefix)) for k in ("mean", "median", "p90", "p99")]
    mx = max(lst) if lst else None
    cells = ["%.1f" % v if isinstance(v, (int, float)) else "-" for v in vals]
    if mx is not None:
        cells.append("%.1f" % mx)
    else:
        cells.append("-")
    print("%-10s %10s %10s %10s %10s %10s" % (label, *cells))

ttfts = [t for t in (g("ttfts") or []) if isinstance(t, (int, float))]
line("TTFT(ms)", "ttft", ttfts)
line("TPOT(ms)", "tpot")
line("ITL(ms)", "itl")
line("E2E(ms)", "e2e_latency")

print("\n吞吐              : output %s tok/s | request %s req/s | input %s tok/s" % (
    g("output_throughput"), g("request_throughput"), g("input_throughput")))
print("平均并发          : %s" % g("concurrency"))

# 完整性
olens = g("output_lens") or []
errs = [e for e in (g("errors") or []) if e]
bad_len = [n for n in olens if n != out_len]
ok_complete = (len(olens) > 0 and not bad_len and not errs)
print("\n完整性            : %s（非满长 %d 条，错误 %d 条）" % ("通过" if ok_complete else "未通过", len(bad_len), len(errs)))
if errs:
    print("   首个错误: %s" % str(errs[0])[:200])

# SLO
def pick(prefix):
    key = {"mean": "mean", "median": "median", "p90": "p90", "p95": "p95", "p99": "p99"}[stat]
    return r.get("%s_%s_ms" % (key, prefix))

t_ttft, t_tpot = pick("ttft"), pick("tpot")
def verdict(v, thr):
    if v is None:
        return "N/A"
    return "PASS" if v <= thr else "FAIL"

print("\nSLO（按 %s 判定）" % stat)
print("  TTFT  %8.1f ms  <= %8.0f ms   %s" % (t_ttft, slo_ttft, verdict(t_ttft, slo_ttft)) if t_ttft is not None else "  TTFT  N/A")
print("  TPOT  %8.1f ms  <= %8.0f ms   %s" % (t_tpot, slo_tpot, verdict(t_tpot, slo_tpot)) if t_tpot is not None else "  TPOT  N/A")

# 命中率校准建议
if hit is not None and abs(hit - cache_pct) > 3:
    print("\n[提示] 实测命中率 %.1f%% 偏离目标 %g%%：" % (hit, cache_pct))
    print("       共享前缀实际占比 = %.1f%%，用 CACHE_PCT=%g 重跑，或手动调 --gsp-system-prompt-len"
          % (hit, cache_pct))

print("\n===== 回传块 =====")
print("HIT=%s%% TARGET=%g%%" % (hit, cache_pct))
print("TTFT_%s=%s ms (SLO %g)" % (stat, t_ttft, slo_ttft))
print("TPOT_%s=%s ms (SLO %g)" % (stat, t_tpot, slo_tpot))
print("OUT_TPUT=%s tok/s  CONC=%s  COMPLETE=%s" % (g("output_throughput"), g("concurrency"), ok_complete))
print("===== 回传块结束 =====")

if not ok_complete:
    sys.exit(1)
if strict and ((t_ttft is not None and t_ttft > slo_ttft) or (t_tpot is not None and t_tpot > slo_tpot)):
    sys.exit(3)
PY
RC=$?
echo
echo "结果文件: ${STEM}.jsonl"
echo "日志文件: ${STEM}.log"
echo "（退出码 $RC：0=完整性通过；1=完整性失败；3=SLO 未达标且 STRICT=1）"
exit $RC
