#!/usr/bin/env bash
# 03 - 干跑 import：确认“跑的是这份代码”且 import 链没坏（H15 检查）。
# 只 import，不加载权重、不建模型实例，所以不需要 MODEL_PATH。
# 用法： bash check_import.sh
set -u

REPO=$(cd "$(dirname "$0")/.." && pwd)
LOG_DIR=${LOG_DIR:-/tmp/dsv41_preflight}
mkdir -p "$LOG_DIR" 2>/dev/null || LOG_DIR=/tmp
LOG="$LOG_DIR/check_import.$(date +%Y%m%d-%H%M%S).log"
VERDICT="$LOG_DIR/import_verdict.env"

echo "===== [1] 代码定位（必须是本仓库）====="
echo "repo=$REPO"
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" log -1 --format='HEAD   : %h %ad %s' --date=short 2>/dev/null || true
  echo "branch : $(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null)"
fi

echo
echo "===== [2] 用本仓库代码 import 关键模块 ====="
TIMEOUT=""
command -v timeout >/dev/null 2>&1 && TIMEOUT="timeout 300"

env PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}" $TIMEOUT \
python3 - "$VERDICT" <<'PY' 2>&1 | tee -a "$LOG"
import sys

verdict_path = sys.argv[1]
res = {}

try:
    import sglang
    res["SGLANG_FILE"] = sglang.__file__
    res["IS_REPO_CODE"] = "NO" if "/site-packages/" in sglang.__file__ else "YES"
    print("sglang.__file__ =", sglang.__file__)
    print("is_repo_code    =", res["IS_REPO_CODE"],
          "" if res["IS_REPO_CODE"] == "YES" else " <-- 注意：import 到了别处的 sglang")
except Exception as e:
    res["SGLANG_FILE"] = "FAIL(%s)" % type(e).__name__
    res["IS_REPO_CODE"] = "UNKNOWN"
    print("import sglang FAIL:", repr(e))

mods = [
    "sglang.srt.models.deepseek_v4",
    "sglang.srt.models.deepseek_v4_platform_stage",
    "sglang.srt.models.deepseek_v4_dspark",
    "sglang.srt.hardware_backend.npu.dsv4.dsv4_rope",
    "sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend",
]
fails = []
for m in mods:
    try:
        __import__(m)
        print("import OK  :", m)
    except Exception as e:
        print("import FAIL:", m, "->", type(e).__name__, e)
        fails.append(m)

# --- 走服务端真实路径：init_npu_backend() 之后再数一次 custom 算子 ---
res["OPS_AFTER_INIT"] = "SKIPPED"
try:
    import torch as _t
    _t.ops  # noqa: B018
    try:
        import torch_npu as _tn  # noqa: F401
        res["TORCH_NPU_PRELOADED"] = "YES"
    except Exception:
        res["TORCH_NPU_PRELOADED"] = "NO"
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend
    init_npu_backend()
    names = ("compressor indexer_compress_epilog inplace_partial_rotary_mul "
             "kv_compress_epilog_v2 npu_hc_post npu_hc_pre_v2 "
             "npu_kv_quant_sparse_attn_sharedkv npu_sparse_attn_sharedkv "
             "npu_moe_gating_top_k npu_quant_lightning_indexer").split()
    try:
        ns = _t.ops.custom
        got = [n for n in names if hasattr(ns, n)]
    except Exception:
        got = []
    print("after init_npu_backend: custom ops %d/%d" % (len(got), len(names)))
    res["OPS_AFTER_INIT"] = "%d/%d" % (len(got), len(names))
except Exception as e:
    print("init_npu_backend 检查失败（可能没有可见 NPU 设备）:", type(e).__name__, e)
    res["OPS_AFTER_INIT"] = "FAIL(%s)" % type(e).__name__

res["IMPORT_FAILS"] = len(fails)
res["IMPORT_FAIL_NAMES"] = ",".join(fails) if fails else "-"

with open(verdict_path, "w") as fh:
    for k, v in res.items():
        fh.write("%s=%s\n" % (k, v))
PY

echo
echo "===== 回传块（复制下面几行）====="
if [ -f "$VERDICT" ]; then
  grep -E '^(IS_REPO_CODE|IMPORT_FAILS|IMPORT_FAIL_NAMES|OPS_AFTER_INIT|TORCH_NPU_PRELOADED|SGLANG_FILE)=' "$VERDICT"
else
  echo "IS_REPO_CODE=UNKNOWN IMPORT_FAILS=UNKNOWN"
fi
echo "REPO_HEAD=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo NA)"
echo "LOG=$LOG"
echo "===== 回传块结束 ====="
