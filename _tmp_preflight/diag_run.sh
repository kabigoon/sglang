#!/usr/bin/env bash
# diag_run.sh —— 一条命令收集「上一轮到底跑到哪」的全部证据，末尾打一个回传块。
# 不解码、不启服务、只读文件，秒级完成。用法：bash _tmp_preflight/diag_run.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SEG=$(cd "$HERE/.." && pwd)
PORT=${PORT:-5684}

hr() { printf '%.0s-' {1..60}; echo; }

echo "===== 回传块（复制整段） ====="
echo "时间: $(date '+%F %T %Z')"
echo "仓库: $(git -C "$SEG" rev-parse --short HEAD 2>/dev/null || echo 非 git)  $(git -C "$SEG" branch --show-current 2>/dev/null)"
echo "工具: python3=$(command -v python3 >/dev/null && echo ok || echo 缺)" \
     "curl=$(command -v curl >/dev/null && echo ok || echo 缺)" \
     "pkill=$(command -v pkill >/dev/null && echo ok || echo 缺)" \
     "pgrep=$(command -v pgrep >/dev/null && echo ok || echo 缺)" \
     "ss=$(command -v ss >/dev/null && echo ok || echo 缺)"
echo "locale: ${LANG:-未设置} / ${LC_ALL:-未设置}"

hr
echo "[1] oneshot 目录（按时间倒序，前 5 个）"
ls -dt /tmp/oneshot_* 2>/dev/null | head -5 | while read -r d; do
  echo "  $d"
  echo "     内容: $(ls -m "$d" 2>/dev/null | tr -d '\n')"
  [ -f "$d/stage.txt" ] && sed 's/^/     stage: /' "$d/stage.txt" | tail -12
done
[ -z "$(ls -d /tmp/oneshot_* 2>/dev/null)" ] && echo "  （没有 oneshot 目录：脚本没执行到 mkdir，或输出目录被清过）"

hr
echo "[2] 最新目录里的关键文件"
D=$(ls -dt /tmp/oneshot_* 2>/dev/null | head -1)
if [ -n "${D:-}" ]; then
  for f in server.log summary.tsv stage.txt; do
    if [ -f "$D/$f" ]; then
      echo "  $f 存在  大小=$(wc -c < "$D/$f")  mtime=$(date -r "$D/$f" '+%F %T' 2>/dev/null || stat -c %y "$D/$f" 2>/dev/null)"
    else
      echo "  $f 不存在"
    fi
  done
  echo "  bench 日志: $(ls "$D"/bench_in*.log 2>/dev/null | tr '\n' ' ')"
  echo
  echo "  --- server.log 里跟就绪/崩溃相关的行（最多 15 行）---"
  grep -nE "v1/models|unbound variable|Command not found|command not found|Killed|Traceback|ERROR|Error" "$D/server.log" 2>/dev/null \
    | tail -15 | cut -c1-160 | sed 's/^/    /'
  echo
  echo "  --- 是否收到过 /v1/models（空=探测一次都没打通）---"
  grep -c "GET /v1/models" "$D/server.log" 2>/dev/null | sed 's/^/    次数=/'
else
  echo "  （无目录）"
fi

hr
echo "[3] bench 产物（bench 自己落盘，与 oneshot 无关）"
ls -lt /home/dsv4_test/bench_out 2>/dev/null | head -4 | sed 's/^/  /'
echo "  今天的: $(find /home/dsv4_test/bench_out -maxdepth 1 -newermt '-1 day' 2>/dev/null | wc -l) 个"

hr
echo "[4] 当前是否还有 server / 端口占用"
pgrep -af 'sglang[.]launch_server' 2>/dev/null | cut -c1-120 | sed 's/^/  /' || true
[ -z "$(pgrep -f 'sglang[.]launch_server' 2>/dev/null)" ] && echo "  （没有 server 在跑）"
python3 - "$PORT" <<'PY' 2>/dev/null | sed 's/^/  /'
import http.client, sys
p = int(sys.argv[1])
try:
    c = http.client.HTTPConnection("127.0.0.1", p, timeout=3)
    c.request("GET", "/v1/models")
    print("port %d /v1/models -> %s" % (p, c.getresponse().status))
except Exception as e:
    print("port %d 不可达: %s" % (p, type(e).__name__))
PY
echo "===== 回传块结束 ====="
