#!/usr/bin/env bash
# 05 - 在本机翻找现成的算子包（无外网时的第一选择）。
# 目标：custom-ops / ops-transformer 的 zip、custom_ops 的 wheel、或已装好的
#       custom_ops 包 / opp venders 目录。找到就不用下载。
# 用法： bash find_artifacts.sh
set -u

echo "===== [1] 搜索常见目录 ====="
ROOTS="/home /root /tmp /mnt /data /opt /workspace /var/tmp"
[ -d /usr/local ] && ROOTS="$ROOTS /usr/local/python3.12.15/lib/python3.12/site-packages"
for r in $ROOTS; do
  [ -d "$r" ] || continue
  timeout 60 find "$r" -maxdepth 6 \
    \( -name 'custom-ops-*.zip' -o -name 'ops-transformer-*.zip' \
       -o -name 'custom_ops-1.0-*.whl' -o -name 'custom_ops-*.whl' \) \
    -print 2>/dev/null
done | sort -u | sed 's/^/  FOUND: /'

echo
echo "===== [2] 已安装的 custom_ops / vendor 目录 ====="
"${PYTHON:-python3}" - <<'PY' 2>/dev/null || true
import glob, os, site
pats = []
try:
    for p in set(list(site.getsitepackages()) + [site.getusersitepackages()]):
        pats += glob.glob(os.path.join(p, "custom_ops*"))
except Exception:
    pass
print("  site-packages custom_ops:", pats or "MISSING")
PY
for d in /usr/local/Ascend/cann-*/opp/vendors /usr/local/Ascend/ascend-toolkit/latest/opp/vendors; do
  [ -d "$d" ] || continue
  echo "  $d -> $(ls -1 "$d" 2>/dev/null | tr '\n' ' ')"
done

echo
echo "===== [3] 本机是否有 docker（有老镜像的话可从里面拷）===="
if command -v docker >/dev/null 2>&1; then
  docker images --format '  {{.Repository}}:{{.Tag}}  {{.Size}}' 2>/dev/null | head -20 || echo "  docker 存在但列不出镜像"
else
  echo "  (当前环境没有 docker CLI —— 大概率你在容器里，这一步要在宿主机做)"
fi

echo
echo "===== 回传块（复制下面几行）====="
N_ZIP=$( { for r in $ROOTS; do [ -d "$r" ] && timeout 60 find "$r" -maxdepth 6 \( -name 'custom-ops-*.zip' -o -name 'ops-transformer-*.zip' \) -print 2>/dev/null; done; } | sort -u | wc -l | tr -d ' ')
N_WHL=$( { for r in $ROOTS; do [ -d "$r" ] && timeout 60 find "$r" -maxdepth 6 -name 'custom_ops-*.whl' -print 2>/dev/null; done; } | sort -u | wc -l | tr -d ' ')
echo "LOCAL_ZIPS=$N_ZIP"
echo "LOCAL_WHEELS=$N_WHL"
echo "如果有 zip：LOCAL_DIR=<上面某个 zip 所在目录> bash _tmp_preflight/fix_install_custom_ops.sh"
echo "===== 回传块结束 ====="
