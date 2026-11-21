#!/usr/bin/env bash
# 01 - 镜像代际 + 算子可用性。只读：不加载权重、不建池、不抓图。
# 用法： bash check_image.sh
set -u

# 允许指定解释器（容器的 python3 未必是跑 sglang 的那个）
PY=${PYTHON:-python3}

REPO=$(cd "$(dirname "$0")/.." && pwd)
LOG_DIR=${LOG_DIR:-/tmp/dsv41_preflight}
mkdir -p "$LOG_DIR" 2>/dev/null || LOG_DIR=/tmp
LOG="$LOG_DIR/check_image.$(date +%Y%m%d-%H%M%S).log"
VERDICT="$LOG_DIR/verdict.env"

# 13 个 PR 分支实际调用的 custom 算子名（顺序无关）
export NAMES="compressor indexer_compress_epilog inplace_partial_rotary_mul \
kv_compress_epilog_v2 npu_hc_post npu_hc_pre_v2 \
npu_kv_quant_sparse_attn_sharedkv npu_kv_quant_sparse_attn_sharedkv_metadata \
npu_moe_gating_top_k npu_quant_lightning_indexer \
npu_quant_lightning_indexer_metadata npu_sparse_attn_sharedkv \
npu_sparse_attn_sharedkv_metadata"

if command -v tee >/dev/null 2>&1; then
  exec > >(tee -a "$LOG") 2>&1
fi

echo "===== [0] 环境 ====="
echo "host=$(hostname 2>/dev/null) pid=$$ date=$(date -Is 2>/dev/null)"
echo "python=$PY -> $(command -v "$PY" 2>/dev/null || echo NOT_FOUND)"
echo "sys.executable=$("$PY" -c "import sys;print(sys.executable)" 2>/dev/null || echo NA)"
echo "sys.prefix=$("$PY" -c "import sys;print(sys.prefix)" 2>/dev/null || echo NA)"
if [ -d "$REPO/.git" ]; then
  echo "repo=$REPO"
  git -C "$REPO" log -1 --format='repo HEAD: %h %ad %s' --date=short 2>/dev/null || true
  git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null | sed 's/^/repo branch: /' || true
else
  echo "repo=$REPO (no .git)"
fi

echo
echo "===== [1] 镜像代际（不需要 NPU 设备）====="
echo "-- pip 关键包 --"
"$PY" -m pip list 2>/dev/null \
  | grep -Ei '^(sgl[-_]kernel[-_]npu|custom[-_]ops|ops[-_]transformer|deep[-_]ep|torch|torch[-_]npu|triton[-_]ascend)[[:space:]]' \
  || echo "(pip 里没有匹配项)"

echo "-- CANN opp vendors --"
ls -1 /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/ 2>/dev/null \
  || ls -1 /usr/local/Ascend/cann-*/opp/vendors/ 2>/dev/null \
  || echo "(vendors 目录不存在)"

echo
echo "===== [1b] source vendor 环境（custom_ops 需要的 aclnn 路径）====="
VENDOR_SOURCED=no
for envf in /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/*/bin/set_env.bash \
            /usr/local/Ascend/cann-*/opp/vendors/*/bin/set_env.bash; do
  [ -f "$envf" ] || continue
  set +u
  # shellcheck disable=SC1090
  . "$envf" >/dev/null 2>&1
  set -u
  echo "  sourced: $envf"
  VENDOR_SOURCED=yes
done
[ "$VENDOR_SOURCED" = "no" ] && echo "  (没有 vendor 的 set_env.bash —— 说明是纯新代镜像，不奇怪)"

echo
echo "===== [2] 算子探测（PR 的 13 个 custom 名字）====="
"$PY" - "$VERDICT" <<'PY'
import glob, os, site, sys

verdict_path = sys.argv[1]
names = os.environ.get("NAMES", "").split()
res = {}

# --- 包是否存在 ---
hits = []
try:
    paths = list(site.getsitepackages()) + [site.getusersitepackages()]
except Exception:
    paths = []
for p in set(paths):
    hits += glob.glob(os.path.join(p, "custom_ops*"))
res["CUSTOM_OPS_PKG"] = "yes" if hits else "no"
print("custom_ops 安装目录:", hits if hits else "MISSING")

try:
    import custom_ops  # noqa: F401
    res["CUSTOM_OPS_IMPORT"] = "OK"
except Exception as e:
    res["CUSTOM_OPS_IMPORT"] = "FAIL(%s)" % type(e).__name__
    if type(e).__name__ == "ImportError":
        print("  提示: 包在但加载失败，通常是这个 shell 没 source vendor 的 set_env.bash")
print("import custom_ops:", res["CUSTOM_OPS_IMPORT"])

# --- torch / torch_npu ---
torch = None
res["OPS_CUSTOM"] = "UNTESTED"
res["OPS_CUSTOM_MISSING"] = "?untested"
res["OPS_NPU"] = "UNTESTED"
try:
    import torch  # noqa: F811
    res["TORCH"] = torch.__version__
except Exception as e:
    res["TORCH"] = "FAIL(%s)" % type(e).__name__
print("torch:", res["TORCH"])
try:
    import torch_npu  # noqa: F401
    res["TORCH_NPU"] = "OK"
except Exception as e:
    res["TORCH_NPU"] = "FAIL(%s)" % type(e).__name__
print("torch_npu:", res["TORCH_NPU"])

# --- 三个命名空间 ---
if torch is not None:
    for ns in ("custom", "npu", "cann_ops_transformer"):
        key = ns.upper()
        try:
            obj = getattr(torch.ops, ns)
        except Exception as e:
            res["OPS_" + key] = "NS_MISSING"
            res["OPS_" + key + "_MISSING"] = "all"
            print("  %-20s 命名空间不存在 (%s)" % (ns, type(e).__name__))
            continue
        got = [n for n in names if hasattr(obj, n)]
        miss = [n for n in names if n not in got]
        res["OPS_" + key] = "%d/%d" % (len(got), len(names))
        res["OPS_" + key + "_MISSING"] = ",".join(miss) if miss else "-"
        print("  %-20s %2d/%d 命中  %s" % (ns, len(got), len(names), ",".join(got) if got else "-"))

# --- 新代 Python API ---
for split in ("rope", "attention", "indexer", "mem_cache", "hyper_conn", "moe"):
    key = "OPERATORS_" + split.upper()
    try:
        __import__("sgl_kernel_npu.operators." + split)
        res[key] = "OK"
    except Exception as e:
        res[key] = "FAIL(%s)" % type(e).__name__
    print("  sgl_kernel_npu.operators.%-11s %s" % (split, res[key]))

# --- 结论 ---
custom_ok = res.get("CUSTOM_OPS_IMPORT") == "OK"
full = res.get("OPS_CUSTOM") == "13/13"
rope_ok = res.get("OPERATORS_ROPE") == "OK"
if custom_ok and full:
    res["VERDICT"] = "OLD_GEN_MATCH"
elif custom_ok:
    res["VERDICT"] = "OLD_GEN_PARTIAL"
elif rope_ok:
    res["VERDICT"] = "NEW_GEN_MISMATCH"
else:
    res["VERDICT"] = "UNKNOWN"

# 理想镜像 = 老代算子齐(13/13) 且 新代 Python API 也在（PR 里 backend 会 import 它）
_need = ["OPERATORS_ROPE", "OPERATORS_ATTENTION", "OPERATORS_INDEXER", "OPERATORS_MEM_CACHE"]
res["IDEAL"] = "YES" if (custom_ok and full and all(res.get(k) == "OK" for k in _need)) else "NO"

with open(verdict_path, "w") as fh:
    for k, v in res.items():
        fh.write("%s=%s\n" % (k, v))
PY

echo
echo "===== 回传块（复制下面这几行）====="
if [ -f "$VERDICT" ]; then
  grep -E '^(CUSTOM_OPS_PKG|CUSTOM_OPS_IMPORT|TORCH|TORCH_NPU|OPS_CUSTOM|OPS_CUSTOM_MISSING|OPS_NPU|OPERATORS_ROPE|OPERATORS_ATTENTION|OPERATORS_INDEXER|OPERATORS_MEM_CACHE|VERDICT|IDEAL)=' "$VERDICT"
  echo "SGL_KERNEL_NPU=$("$PY" -m pip show sgl-kernel-npu 2>/dev/null | awk '/^Version:/ {print $2}')"
  echo "VENDOR_ENV_SOURCED=$VENDOR_SOURCED"
  echo "LOG=$LOG"
else
  echo "VERDICT=NO_OUTPUT  (python 探测失败，把上面整段日志发回)"
fi
echo "===== 回传块结束 ====="
