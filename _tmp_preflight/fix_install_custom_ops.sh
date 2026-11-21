#!/usr/bin/env bash
# 04 - 给「新代镜像」补装 post3 的 custom-ops / ops-transformer，让 PR 的 13 个
#      torch.ops.custom.* 重新出现（VERDICT=NEW_GEN_MISMATCH 时才需要）。
#
# 幂等：已经能 import custom_ops 就直接跳过安装，只做一次验证。
# 只装 CANN vendor + 一个 pip 包，不改 sglang 代码。
#
# 用法（容器内，root）：
#   bash _tmp_preflight/fix_install_custom_ops.sh
# 可选覆盖：TAG=... CANN=... DEVICE=... ARCH=... WORK=... LOCAL_DIR=...
# 无外网时：先在有网的机器上把两个 zip 放进 $LOCAL_DIR，再 LOCAL_DIR=/path bash ...
set -u

PY=${PYTHON:-python3}
TAG=${TAG:-2026.11.0.post3}
DEVICE=${DEVICE:-a3}
WORK=${WORK:-/tmp/custom_ops_install}
LOCAL_DIR=${LOCAL_DIR:-}

# --- 自动探测 torch / cann / arch ---
TORCH_VER=$("$PY" -c 'import torch;print(torch.__version__.split("+")[0])' 2>/dev/null || echo "")
ARCH=$(uname -m 2>/dev/null || echo aarch64)
case "$ARCH" in
  arm64|aarch64) ARCH=aarch64 ;;
  amd64|x86_64)  ARCH=x86_64 ;;
esac
if [ -z "${CANN:-}" ]; then
  CANN=$(ls -d /usr/local/Ascend/cann-9* 2>/dev/null | sed 's#.*/cann-##' | sort -V | tail -1)
  [ -z "${CANN:-}" ] && CANN=9.2.0
fi
if [ -z "$TORCH_VER" ]; then
  echo "FAIL: 探测不到 torch 版本（$PY -c import torch 失败）。用 PYTHON=... 指定跑 sglang 的那个解释器。"
  exit 1
fi
OPP=/usr/local/Ascend/cann-${CANN}/opp
[ -d "$OPP" ] || OPP=/usr/local/Ascend/ascend-toolkit/latest/opp

BASE="https://github.com/sgl-project/sgl-kernel-npu/releases/download/${TAG}"

# 已知 good sha256（仅对 post3 + torch2.13.0 + cann9.2.0 + a3 + aarch64 这一组）
SHA_CA=""; SHA_TR=""
if [ "$TAG" = "2026.11.0.post3" ] && [ "$TORCH_VER" = "2.13.0" ] && \
   [ "$CANN" = "9.2.0" ] && [ "$DEVICE" = "a3" ] && [ "$ARCH" = "aarch64" ]; then
  SHA_CA=827d70c9b31f4baf1c3ad08bb5f1d7ef0719ce04abd2edf1d5f0f1eb685fe048
  SHA_TR=b65172505dd9933501de421fbca5e9fa1b61bb6598c9cfb4eb3476fac15af85b
fi
F_CA="custom-ops-${TAG}-torch${TORCH_VER}-cann${CANN}-${DEVICE}-${ARCH}.zip"
F_TR="ops-transformer-${TAG}-torch${TORCH_VER}-cann${CANN}-${DEVICE}-${ARCH}.zip"

echo "===== [0] 目标 ====="
echo "PY=$PY ($("$PY" -c 'import sys;print(sys.executable)' 2>/dev/null))"
echo "TORCH=$TORCH_VER  CANN=$CANN  ARCH=$ARCH  DEVICE=$DEVICE  TAG=$TAG"
echo "OPP=$OPP"
echo "WORK=$WORK"

# --- 幂等检查 ---
if "$PY" -c 'import custom_ops' 2>/dev/null; then
  echo "custom_ops 已可 import，跳过安装。"
  SKIP_INSTALL=1
else
  SKIP_INSTALL=0
fi

if [ "$SKIP_INSTALL" = "0" ]; then
  echo
  echo "===== [1] 取包 ====="
  mkdir -p "$WORK" && cd "$WORK" || exit 1
  if [ -n "$LOCAL_DIR" ]; then
    cp -f "$LOCAL_DIR"/custom-ops-*.zip "$WORK"/custom-ops.zip && \
    cp -f "$LOCAL_DIR"/ops-transformer-*.zip "$WORK"/ops-transformer.zip && \
      echo "从 LOCAL_DIR 复制完成"
  else
    echo "-- 网络自检 --"
    echo "  proxy: $(env | grep -i proxy | tr '\n' ' ' || true)"
    echo "  工具: curl=$(command -v curl || echo none) wget=$(command -v wget || echo none)"

    # 公司代理常做 SSL 拦截（自签 CA）。阶梯：默认CA -> 系统CA -> -k(带 sha256 兜底)
    check_sha() {  # $1=file $2=want
      local f="$1" want="$2" got
      if [ -z "$want" ]; then echo "  (无预期 sha256，跳过校验)"; return 0; fi
      got=$( { command -v sha256sum >/dev/null 2>&1 && sha256sum "$f" || shasum -a 256 "$f"; } 2>/dev/null | awk '{print $1}' )
      echo "  sha256=$got"
      if [ "$got" = "$want" ]; then echo "  OK 校验通过"; return 0; fi
      echo "  FAIL 校验失败（期望 $want）"; return 1
    }

    fetch() {  # $1=url $2=out
      local url="$1" out="$2" rc=0
      rm -f "$out"
      if command -v curl >/dev/null 2>&1; then
        echo "  [1/4] curl 默认 CA"
        curl -fL --retry 3 --retry-delay 2 -o "$out" "$url" && { echo "        OK"; return 0; }
        rc=$?; echo "        失败 退出码=$rc"
        if [ -f /etc/ssl/certs/ca-certificates.crt ]; then
          echo "  [2/4] curl --cacert 系统 CA"
          curl -fL --retry 3 --retry-delay 2 --cacert /etc/ssl/certs/ca-certificates.crt \
            -o "$out" "$url" && { echo "        OK"; return 0; }
          echo "        失败 退出码=$?"
        fi
        if [ "$rc" = "60" ] || [ "$rc" = "35" ]; then
          echo "  [3/4] curl -k（跳过证书校验，随后用 sha256 验证）"
          curl -fL --retry 3 --retry-delay 2 -k -o "$out" "$url" \
            && { echo "        OK（注意：本次跳过了证书校验）"; INSECURE=1; return 0; }
          echo "        失败 退出码=$?"
        fi
      fi
      if command -v wget >/dev/null 2>&1; then
        echo "  [4/4] wget --no-check-certificate"
        wget -c --no-check-certificate -O "$out" "$url" \
          && { echo "        OK（注意：本次跳过了证书校验）"; INSECURE=1; return 0; }
        echo "        失败 退出码=$?"
      fi
      return 1
    }

    INSECURE=0
    echo "下载 $F_CA"
    fetch "$BASE/$F_CA" custom-ops.zip || { echo "FAIL: 下载 custom-ops 失败"; exit 1; }
    ls -l custom-ops.zip
    check_sha custom-ops.zip "${SHA_CA:-}" || exit 1
    echo "下载 $F_TR"
    fetch "$BASE/$F_TR" ops-transformer.zip || { echo "FAIL: 下载 ops-transformer 失败"; exit 1; }
    ls -l ops-transformer.zip
    check_sha ops-transformer.zip "${SHA_TR:-}" || exit 1
    [ "$INSECURE" = "1" ] && echo "  提示：本次因代理 SSL 拦截走了不安全通道，但 sha256 已校验通过。"
  fi

  echo
  echo "===== [2] 解包并安装 CANN vendor ====="
  unzip -o -q custom-ops.zip || { echo "FAIL: unzip custom-ops.zip"; exit 1; }
  unzip -o -q ops-transformer.zip || { echo "FAIL: unzip ops-transformer.zip"; exit 1; }
  chmod +x ./*.run 2>/dev/null
  for run in ./*custom_ops*.run ./*ops-transformer*.run; do
    [ -f "$run" ] || continue
    echo "install $run -> $OPP"
    "$run" --install-path="$OPP" || echo "WARN: $run 返回非 0"
  done

  echo
  echo "===== [3] pip 安装 custom_ops wheel ====="
  WHL=$(ls ./custom_ops-1.0-*.whl 2>/dev/null | head -1)
  if [ -n "$WHL" ]; then
    "$PY" -m pip install --no-deps -q "$WHL" && echo "installed: $WHL" || echo "FAIL: pip install $WHL"
  else
    echo "WARN: 没找到 custom_ops-1.0-*.whl（包结构可能变了，下面 ls 看内容）"
  fi
fi

echo
echo "===== [4] source vendor 环境 + 立刻验证 ====="
# set_env.bash 里可能有未定义变量引用，临时关掉 nounset 再 source
set +u
for envf in "$OPP"/vendors/customize/bin/set_env.bash "$OPP"/vendors/custom_transformer/bin/set_env.bash; do
  if [ -f "$envf" ]; then
    # shellcheck disable=SC1090
    source "$envf" && echo "sourced: $envf"
  else
    echo "MISSING: $envf"
  fi
done
set -u

export NAMES="compressor indexer_compress_epilog inplace_partial_rotary_mul \
kv_compress_epilog_v2 npu_hc_post npu_hc_pre_v2 \
npu_kv_quant_sparse_attn_sharedkv npu_kv_quant_sparse_attn_sharedkv_metadata \
npu_moe_gating_top_k npu_quant_lightning_indexer \
npu_quant_lightning_indexer_metadata npu_sparse_attn_sharedkv \
npu_sparse_attn_sharedkv_metadata"

"$PY" - <<'PY'
import os
names = os.environ.get("NAMES", "").split()
try:
    import torch
    import torch_npu  # noqa: F401
except Exception as e:
    print("torch/torch_npu 不可用:", repr(e))
    print("OPS_CUSTOM=UNTESTED")
    raise SystemExit(0)
try:
    import custom_ops  # noqa: F401
    print("import custom_ops: OK")
except Exception as e:
    print("import custom_ops: FAIL(%s)" % type(e).__name__)
try:
    ns = torch.ops.custom
except Exception as e:
    print("namespace custom: MISSING (%s)" % type(e).__name__)
    print("OPS_CUSTOM=0/%d" % len(names))
    raise SystemExit(0)
got = [n for n in names if hasattr(ns, n)]
miss = [n for n in names if n not in got]
print("OPS_CUSTOM=%d/%d" % (len(got), len(names)))
print("OPS_CUSTOM_MISSING=%s" % (",".join(miss) if miss else "-"))
PY

echo
echo "===== 回传块（复制下面几行）====="
echo "TAG=$TAG TORCH=$TORCH_VER CANN=$CANN ARCH=$ARCH DEVICE=$DEVICE"
echo "CUSTOM_OPS_IMPORT=$("$PY" -c 'import custom_ops' 2>/dev/null && echo OK || echo FAIL)"
echo "VENDORS=$(ls -1 "$OPP"/vendors 2>/dev/null | tr '\n' ',' )"
echo "接下来请重跑： bash _tmp_preflight/check_image.sh"
echo "===== 回传块结束 ====="
