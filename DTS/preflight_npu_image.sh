#!/bin/bash
# Pre-flight: does this NPU image match the PR #50205 code on the host?
# Runs INSIDE the container. Read-only: imports packages, reads dirs, no weights, no graph.
set -u

echo "===== [0] 挂载进来的代码 ====="
if [ -d /repo/.git ]; then
  git -C /repo log -1 --format='repo HEAD: %h %ad %s' --date=short 2>/dev/null || echo "repo: git 读取失败"
else
  echo "repo: 未挂载（跳过）"
fi

echo
echo "===== [1] 镜像代际（不需要 NPU 设备）====="
python3 -m pip list 2>/dev/null \
  | grep -Ei '^(sgl[-_]kernel[-_]npu|custom[-_]ops|ops[-_]transformer|deep[-_]ep|torch|torch[-_]npu|triton[-_]ascend)[[:space:]]' \
  || echo "(pip 里没有匹配项)"

echo "-- site-packages 中的 custom_ops --"
python3 - <<'PY'
import glob, os, site
try:
    paths = list(site.getsitepackages()) + [site.getusersitepackages()]
except Exception:
    paths = []
hits = []
for p in set(paths):
    hits += glob.glob(os.path.join(p, "custom_ops*"))
print("custom_ops:", hits if hits else "MISSING")
PY

echo "-- CANN opp vendors --"
ls -1 /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/ 2>/dev/null || echo "(目录不存在)"

echo
echo "===== [2] 算子可用性：PR 分支实际调用的 13 个 custom 名字 ====="
python3 - <<'PY'
NAMES = [
    "compressor", "indexer_compress_epilog", "inplace_partial_rotary_mul",
    "kv_compress_epilog_v2", "npu_hc_post", "npu_hc_pre_v2",
    "npu_kv_quant_sparse_attn_sharedkv", "npu_kv_quant_sparse_attn_sharedkv_metadata",
    "npu_moe_gating_top_k", "npu_quant_lightning_indexer",
    "npu_quant_lightning_indexer_metadata", "npu_sparse_attn_sharedkv",
    "npu_sparse_attn_sharedkv_metadata",
]
try:
    import torch
except Exception as e:
    print("torch 不可用，跳过算子探测:", repr(e))
    raise SystemExit(0)
try:
    import torch_npu  # noqa: F401
    npu_state = "import OK"
except Exception as e:
    npu_state = "import FAIL (%s)" % type(e).__name__
print("torch:", torch.__version__, "| torch_npu:", npu_state)
for ns in ("custom", "npu", "cann_ops_transformer"):
    try:
        obj = getattr(torch.ops, ns)
    except Exception as e:
        print("  namespace %-20s MISSING (%s)" % (ns, type(e).__name__))
        continue
    got = [n for n in NAMES if hasattr(obj, n)]
    print("  namespace %-20s %2d/%d  %s" % (ns, len(got), len(NAMES), ",".join(got) if got else "-"))
PY

echo
echo "===== [3] 新代 API 是否存在（决定能否走“移植 #50203”路线）====="
python3 - <<'PY'
for split in ("rope", "attention", "indexer", "mem_cache", "hyper_conn", "moe"):
    try:
        __import__("sgl_kernel_npu.operators." + split)
        print("  sgl_kernel_npu.operators.%-10s OK" % split)
    except Exception as e:
        print("  sgl_kernel_npu.operators.%-10s %s" % (split, type(e).__name__))
PY

echo
echo "===== 结论 ====="
if python3 -c 'import custom_ops' 2>/dev/null; then
  echo "RESULT: 含 custom_ops（老代镜像）-> 与 PR 代码同代；仍要看 [2] 里 custom 是否 13/13"
elif python3 -c 'import sgl_kernel_npu' 2>/dev/null; then
  echo "RESULT: 有 sgl_kernel_npu 但无 custom_ops（新代镜像）-> PR 的 13 个 torch.ops.custom 调用必然 AttributeError"
else
  echo "RESULT: 两者都没有 -> 这个镜像里根本没有 NPU kernel 包，先确认镜像是否拿对了"
fi
