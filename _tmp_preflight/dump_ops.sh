#!/usr/bin/env bash
# 02 - 把三个命名空间里“已注册的算子名”全量导出（check_image.sh 部分命中时才需要）。
# 用法： bash dump_ops.sh        # 完整清单写 /tmp，屏幕只留摘要
set -u

LOG_DIR=${LOG_DIR:-/tmp/dsv41_preflight}
mkdir -p "$LOG_DIR" 2>/dev/null || LOG_DIR=/tmp
DUMP="$LOG_DIR/ops_dump.$(date +%Y%m%d-%H%M%S).txt"

echo "===== 导出已注册算子 -> $DUMP ====="
python3 - "$DUMP" <<'PY'
import sys

out = open(sys.argv[1], "w")
def emit(*a):
    print(*a)
    print(*a, file=out)

KEY = ("rope", "rotary", "compress", "indexer", "attn", "attention", "hc_",
       "kv", "gating", "swiglu", "quant", "routing", "matmul", "sparse")

counts = {}
try:
    import torch
    import torch_npu  # noqa: F401
except Exception as e:
    emit("torch/torch_npu 不可用:", repr(e))
    emit("===== 回传块（复制下面几行）=====")
    emit("CUSTOM_TOTAL=UNTESTED")
    emit("NPU_TOTAL=UNTESTED")
    emit("TORCH=FAIL(%s)" % type(e).__name__)
    emit("DUMP=%s" % sys.argv[1])
    out.close()
    raise SystemExit(0)

for ns in ("custom", "npu", "cann_ops_transformer"):
    try:
        obj = getattr(torch.ops, ns)
    except Exception as e:
        counts[ns] = "NS_MISSING"
        emit("\n### %s: 命名空间不存在 (%s)" % (ns, type(e).__name__))
        continue
    names = sorted(n for n in dir(obj) if not n.startswith("_"))
    counts[ns] = len(names)
    emit("\n### %s (%d) ###" % (ns, len(names)))
    for n in names:
        emit("   ", n)

emit("\n### sgl_kernel_npu.operators 子模块 ###")
try:
    import sgl_kernel_npu.operators as ops
    subs = sorted(n for n in dir(ops) if not n.startswith("_"))
    emit("   ", ",".join(subs))
except Exception as e:
    emit("    不可用:", repr(e))

# 只把“和 DSV4 相关”的名字挑出来放回传块，避免整份贴
emit("\n===== 回传块（复制下面几行）=====")
for ns in ("custom", "npu", "cann_ops_transformer"):
    try:
        obj = getattr(torch.ops, ns)
        names = [n for n in dir(obj) if not n.startswith("_")]
    except Exception:
        emit("%s=NS_MISSING" % ns.upper())
        continue
    emit("%s_TOTAL=%d" % (ns.upper(), len(names)))
    picked = [n for n in names if any(k in n.lower() for k in KEY)]
    emit("%s_DSV4_RELATED=%s" % (ns.upper(), ",".join(sorted(picked)) or "-"))
emit("DUMP=%s" % sys.argv[1])

out.close()
PY

echo
echo "（完整清单在上面打印的 DUMP 路径里；如果只想要摘要，上面回传块已经够了）"
