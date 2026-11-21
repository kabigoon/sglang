#!/usr/bin/env bash
# 06 - 从宿主机检查任意镜像（不用先 enter 容器）。
# 用法： bash _tmp_preflight/check_any_image.sh <镜像名或ID> [解释器路径]
# 例：   bash _tmp_preflight/check_any_image.sh sglang-npu:2026.11.10
#        bash _tmp_preflight/check_any_image.sh my-img /usr/local/python3.12.15/bin/python3
set -u

IMG=${1:-}
PYARG=${2:-}

if [ -z "$IMG" ]; then
  echo "用法: bash _tmp_preflight/check_any_image.sh <镜像名或ID> [解释器路径]"
  exit 2
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "FAIL: 当前环境没有 docker CLI（你可能在容器里面）。"
  echo "      进到目标容器后改跑： bash _tmp_preflight/check_image.sh"
  exit 2
fi

REPO=$(cd "$(dirname "$0")/.." && pwd)
echo "image : $IMG"
echo "repo  : $REPO (只读挂到容器 /repo)"
echo "python: ${PYARG:-镜像默认 python3}"
echo

E_OPT=""
[ -n "$PYARG" ] && E_OPT="-e PYTHON=$PYARG"

docker run --rm -v "$REPO":/repo:ro $E_OPT --entrypoint bash "$IMG" \
  /repo/_tmp_preflight/check_image.sh
RC=$?

echo
echo "（退出码 ${RC}；如果这里报 image not found，先 docker images | grep -i npu 确认 tag）"
