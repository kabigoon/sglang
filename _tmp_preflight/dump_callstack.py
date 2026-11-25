#!/usr/bin/env python3
"""dump_callstack.py —— 从 trace_view.json 里抽「算子 → Python 调用栈」，把 kernel 对回代码。

为什么要它：`kernel_details.csv` 只有算子名/形状/pipe 比率，**没有栈**；"这个 kernel 是谁
在 Python 侧发起的"只能从 `trace_view.json` 的 Python 栈事件里取。这个脚本把动辄几百 MB
的 trace 压成几 KB 的 markdown 对照表，可以直接提交回仓库做证据。

用法（在 NPU 机器上，仓库根目录执行；换成实际 trace 目录）：
  # 0) 先探一下这个版本的 trace 长什么样（字段名/有没有栈），不行再看提示
  python3 _tmp_preflight/dump_callstack.py /tmp/demo_ep_late_1124_135951 --probe

  # 1) 默认清单（KERNEL_CODE_MAP 里所有"可疑/待确认"的算子）+ 写 markdown
  python3 _tmp_preflight/dump_callstack.py /tmp/demo_ep_late_1124_135951 --tag demo_ep_late

  # 2) 只想查某几个算子（正则，忽略大小写，子串匹配）
  python3 _tmp_preflight/dump_callstack.py <dir> --ops-re 'SparseLightningIndexer' --ops-re 'LeftShift'

  # 3) 不知道查谁：按总耗时自动取前 15 个算子
  python3 _tmp_preflight/dump_callstack.py <dir> --top 15

  # 4) 顺带看 Python 泳道（forward 骨架 / nn.Module 层级）
  python3 _tmp_preflight/dump_callstack.py <dir> --ops-re 'nn.Module: VocabParallelEmbedding' --ops-re 'forward'

输出：默认写到 `_tmp_preflight/profile_bundle/callstacks/<tag>.md`（几 KB，可 commit）。
只读输入，不碰任何 python/sglang/** 文件。
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import socket
import statistics
import sys
import time
from collections import Counter, defaultdict

try:  # 大文件流式解析（可选依赖）
    import ijson  # type: ignore
except Exception:  # pragma: no cover
    ijson = None

# ---------------------------------------------------------------- 常量

# 判定"这是仓库代码帧"的特征串（用于挑 launcher）
REPO_HINTS = (
    "sglang/",
    "sgl_kernel_npu",
    "sgl-kernel-npu",
    "deep_ep/",
    "ops-transformer",
)

# torch 自己插入的"最内层"帧特征，用来判断栈的方向（inner-first vs inner-last）
TORCH_INNER_HINTS = (
    "torch/_ops.py",
    "torch/nn/modules/module.py",
    "torch/autograd/",
    "torch/_tensor.py",
    "contextlib.py",
)

# 栈可能挂在这些 args key 上（大小写不敏感）
STACK_KEYS = (
    "call stack",
    "callstack",
    "call_stack",
    "python stack",
    "python_stack",
    "python call stack",
    "stack",
    "py stack",
    "frames",
)

# kernel 事件与携带栈的 cpu_op 事件之间常用的关联 id
ID_KEYS = (
    "external id",
    "external_id",
    "correlation",
    "correlation_id",
    "linked id",
    "linked_id",
)

# 默认清单：来自 _tmp_preflight/KERNEL_CODE_MAP.md 的"吃时间 / 待确认"算子。
# 注意：elementwise 那几个用 `(?:^|_)X(?:_|$)` 锚定，否则 `Mul` 会把 `BatchMatMul` 也抓进来。
DEFAULT_OPS = (
    # --- indexer（第一梯队）---
    ("SparseLightningIndexer", "SparseLightningIndexer"),
    ("LightningIndexerV2", "LightningIndexerV2"),
    # --- attention ---
    ("SparseFlashMla", "SparseFlashMla"),
    ("SparseFlashMlaMetadata", "SparseFlashMlaMetadata"),
    # --- MoE 通信 ---
    ("CamMoeCombineNormal", "CamMoeCombineNormal"),
    ("CamMoeDispatchNormal", "CamMoeDispatchNormal"),
    ("NotifyDispatch", "NotifyDispatch"),
    ("DispatchLayout", "DispatchLayout"),
    # --- 待确认来源的小算子 ---
    ("_hc_broadcast_kernel", "_hc_broadcast_kernel"),
    ("LeftShift", "LeftShift"),
    ("RepeatInterleave", "RepeatInterleave"),
    # --- 大 matmul ---
    ("BatchMatMulV2", "BatchMatMulV2"),
    ("MatMulV3", "MatMulV3"),
    ("GroupedMatmul", "GroupedMatmul"),
    ("QuantBatchMatmulV3", "QuantBatchMatmulV3"),
    # --- 前向锚点（每次 forward 恰一次，用来切步）---
    ("GatherV2", "GatherV2"),
    # --- elementwise 聚合（栈会很杂，默认只保留前几个）---
    ("Cast", r"(?:^|_)Cast(?:_|$)"),
    ("ReduceSum", r"(?:^|_)ReduceSum(?:_|$)"),
    ("RealDiv", r"(?:^|_)RealDiv(?:_|$)"),
    ("Add/Adds", r"(?:^|_)Adds?(?:_|$)"),
    ("Mul", r"(?:^|_)Mul(?:_|$)"),
)

# 帧解析：兼容 "file.py(123): func" / "func (file.py:123)" / "file.py:123"
_FRAME_RES = (
    re.compile(r"^(?P<f>.+?)\((?P<l>\d+)\)\s*:\s*(?P<n>.+)$"),
    re.compile(r"^(?P<n>\S+)\s*\((?P<f>[^()]+?):(?P<l>\d+)\)$"),
    re.compile(r"^(?P<f>[\w./\-+]+\.py):(?P<l>\d+)(?:\s+in\s+(?P<n>\S+))?$"),
)


# ---------------------------------------------------------------- 输入定位


def _open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _human_mb(n: float) -> str:
    if n < 1.0:
        return "%.0f KB" % (n * 1024)
    if n < 1024:
        return "%.1f MB" % n
    return "%.2f GB" % (n / 1024)


def find_trace(path: str) -> str:
    """文件直接用；目录则在里面找最像 trace 的那份 json。"""
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        sys.exit("输入不存在：%s" % path)

    cands: list[str] = []
    for pat in (
        "**/ASCEND_PROFILER_OUTPUT/trace_view*.json*",
        "**/trace_view*.json*",
        "**/*profiler*/*.json*",
        "**/*.json*",
    ):
        cands = glob.glob(os.path.join(path, pat), recursive=True)
        if cands:
            break
    cands = [c for c in cands if os.path.getsize(c) > 1024]
    if not cands:
        sys.exit("在 %s 下没找到 trace_view*.json*（是不是只拷了 kernel_details.csv？）" % path)

    def rank(c: str):
        base = os.path.basename(c)
        return ("trace_view" in base, "ASCEND_PROFILER_OUTPUT" in c, os.path.getsize(c))

    return max(cands, key=rank)


class Trace:
    """尽量省内存地遍历 traceEvents：小文件一次读进内存，大文件（有 ijson 时）流式。"""

    def __init__(self, path: str, big_mb: float):
        self.path = path
        self.size = os.path.getsize(path)
        self.stream = self.size > big_mb * 1024 * 1024 and ijson is not None
        self._cache: list | None = None
        self.top_keys: list[str] = []

    def events(self):
        if self._cache is not None:
            yield from self._cache
            return
        if self.stream:
            with _open_text(self.path) as f:
                for ev in ijson.items(f, "traceEvents.item"):
                    if isinstance(ev, dict):
                        yield ev
            return
        with _open_text(self.path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            self.top_keys = list(data.keys())
            evs = data.get("traceEvents") or data.get("events") or []
        else:
            self.top_keys = ["<bare list>"]
            evs = data
        self._cache = [e for e in evs if isinstance(e, dict)]
        yield from self._cache


# ---------------------------------------------------------------- 栈解析


def _get_ci(d: dict, keys) -> object | None:
    """大小写不敏感地取 dict 里的一个 key。"""
    if not isinstance(d, dict):
        return None
    low = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        if k in low:
            return low[k]
    return None


def parse_frames(raw) -> list[dict]:
    """把各种形态的栈归一成 [{file,line,name,raw}]，**保持原始顺序**。"""
    if raw is None:
        return []
    items: list = []
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        parts = re.split(r"\n|;\s*|<-|\|\|", s)
        items = [p.strip() for p in parts if p.strip()]
    elif isinstance(raw, (list, tuple)):
        for x in raw:
            if isinstance(x, str):
                items.append(x.strip())
            elif isinstance(x, dict):
                items.append(x)
    elif isinstance(raw, dict):
        items = [raw]
    out: list[dict] = []
    for it in items:
        if isinstance(it, dict):
            f = it.get("file") or it.get("filename") or it.get("path") or ""
            ln = it.get("line") or it.get("lineno") or 0
            n = it.get("name") or it.get("func") or it.get("function") or ""
            out.append({"file": str(f), "line": int(ln or 0), "name": str(n), "raw": ""})
            continue
        s = str(it).strip()
        if not s:
            continue
        for rx in _FRAME_RES:
            m = rx.match(s)
            if m:
                out.append(
                    {
                        "file": m.group("f").strip(),
                        "line": int(m.group("l")),
                        "name": (m.groupdict().get("n") or "").strip(),
                        "raw": s,
                    }
                )
                break
        else:
            out.append({"file": "", "line": 0, "name": s, "raw": s})
    return out


def event_stack(ev: dict) -> list[dict]:
    args = ev.get("args")
    raw = _get_ci(args, STACK_KEYS) if isinstance(args, dict) else None
    if raw is None:
        raw = _get_ci(ev, STACK_KEYS)  # 少数导出把栈放在顶层
    return parse_frames(raw)


def event_id(ev: dict):
    args = ev.get("args") if isinstance(ev.get("args"), dict) else {}
    for k in ID_KEYS:
        v = _get_ci(args, (k,))
        if v is not None:
            return str(v)
    if isinstance(ev.get("id"), (str, int)):
        return str(ev["id"])
    return None


def _blob(fr: dict) -> str:
    return ((fr.get("file") or "") + " " + (fr.get("raw") or "")).replace("\\", "/")


def is_repo(fr: dict) -> bool:
    b = _blob(fr)
    return any(h in b for h in REPO_HINTS)


def is_sglang(fr: dict) -> bool:
    return "sglang/" in _blob(fr) or "/sglang/srt/" in _blob(fr)


def _is_inner_torch(fr: dict) -> bool:
    b = _blob(fr)
    return any(h in b for h in TORCH_INNER_HINTS)


def normalize_order(frames: list[dict]) -> tuple[list[dict], str]:
    """统一成「最内层在前」。判据：torch 的 dispatch 帧（_ops/module.py）在哪一端。"""
    if len(frames) < 2:
        return frames, "n/a"
    first_inner, last_inner = _is_inner_torch(frames[0]), _is_inner_torch(frames[-1])
    if last_inner and not first_inner:
        return list(reversed(frames)), "inner-last（已反转成最内层在前）"
    if first_inner and not last_inner:
        return frames, "inner-first（本来就是最内层在前）"
    return frames, "未确定（按原始顺序打印）"


def short(path: str, keep: int = 2) -> str:
    if not path:
        return ""
    p = path.replace("\\", "/")
    parts = [x for x in p.split("/") if x and x != "."]
    if len(parts) <= keep:
        return "/".join(parts)
    return ".../" + "/".join(parts[-keep:])


def fmt_frame(fr: dict) -> str:
    f = short(fr.get("file") or "")
    n = fr.get("name") or ""
    ln = fr.get("line") or 0
    if f and ln:
        return ("%s:%s  %s" % (f, ln, n)) if n else ("%s:%s" % (f, ln))
    return fr.get("raw") or n or "?"


def repo_chain(frames: list[dict], limit: int = 4) -> str:
    """栈里所有"仓库代码帧"，从最内层往外，最多 limit 个 —— 这就是可读的调用链。"""
    inner, _ = normalize_order(frames)
    hit = [fmt_frame(fr) for fr in inner if is_repo(fr)]
    if not hit:
        return ""
    shown = hit[:limit]
    tail = "" if len(hit) <= limit else " ⟵ …(共 %d 个仓库帧)" % len(hit)
    return " ⟵ ".join(shown) + tail


def launcher_sglang(frames: list[dict]) -> str:
    inner, _ = normalize_order(frames)
    for fr in inner:
        if is_sglang(fr):
            return fmt_frame(fr)
    return ""


def launcher_any(frames: list[dict]) -> str:
    inner, _ = normalize_order(frames)
    for fr in inner:
        if is_repo(fr):
            return fmt_frame(fr)
    return ""


def frames_key(frames: list[dict]) -> str:
    inner, _ = normalize_order(frames)
    return "|".join(
        "%s:%s:%s" % (short(fr.get("file") or "", 3), fr.get("line") or 0, fr.get("name") or "")
        for fr in inner
    )


def build_span_index(spans: list[tuple[float, float, str]]):
    """按 start 排序并抽出 starts 数组，供二分查找（只做一次）。"""
    spans.sort(key=lambda x: x[0])
    return [sp[0] for sp in spans], spans


def enclosing_py_stack(starts, spans, ts: float, look_back: int = 96):
    """在 python span 里找包含 ts 的、最内层（dur 最小）的那个，返回 stack_key。

    注意：这是「时间包含」推断。kernel 事件的时间是 device 执行时间，与 host 下发时间
    有队列偏移，所以异步下发重的时候可能错配 —— 调用方应把结果标成「推断」。
    """
    import bisect

    if not spans:
        return None
    i = bisect.bisect_right(starts, ts) - 1
    if i < 0:
        return None
    best, best_dur = None, None
    j, n = i, 0
    while j >= 0 and n < look_back:
        st, en, k = spans[j]
        # 只回看 1 小时（trace 的时间单位是 us），且必须真的包含 ts
        if en >= ts and (ts - st) <= 3_600_000_000:
            d = en - st
            if best_dur is None or d < best_dur:
                best, best_dur = k, d
        j -= 1
        n += 1
    return best


# ---------------------------------------------------------------- 主流程


def main() -> int:
    ap = argparse.ArgumentParser(
        description="从 trace_view.json 抽算子 → Python 调用栈",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("trace", help="trace_view.json / .json.gz / 采集目录（含 *_ascend_pt）")
    ap.add_argument(
        "--out",
        default=None,
        help="输出 markdown（默认 _tmp_preflight/profile_bundle/callstacks/<tag>.md）",
    )
    ap.add_argument("--tag", default=None, help="输出文件名用的 tag（默认取输入目录名）")
    ap.add_argument(
        "--ops-re",
        action="append",
        default=[],
        help="算子名正则（可多次，忽略大小写、子串匹配）；给了就替代默认清单",
    )
    ap.add_argument("--top", type=int, default=0, help="不用清单，按总耗时自动取前 N 个算子")
    ap.add_argument("--per-op", type=int, default=400, help="每个算子最多抽多少个事件（去重前）")
    ap.add_argument("--stacks", type=int, default=4, help="每个算子最多打印几种去重后的栈")
    ap.add_argument("--frames", type=int, default=12, help="每种栈最多打印多少帧（最内层在前）")
    ap.add_argument("--big-mb", type=float, default=400.0, help="超过这个大小就尝试 ijson 流式解析")
    ap.add_argument(
        "--infer-by-time",
        action="store_true",
        help="kernel 事件没有栈时，用「时间包含」找最内层 python scope 的栈（异步下可能错配，输出会标 [推断]）",
    )
    ap.add_argument("--probe", action="store_true", help="只做体检：字段名/有没有栈/事件分布")
    args = ap.parse_args()

    path = find_trace(args.trace)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print("trace: %s (%s)" % (path, _human_mb(size_mb)), file=sys.stderr)
    if size_mb > args.big_mb and ijson is None:
        print(
            "  [warn] 文件较大且没装 ijson，会整份 json.load（内存 ~3-5x 文件大小）。"
            "装 ijson 可流式解析：pip install ijson",
            file=sys.stderr,
        )
    trace = Trace(path, args.big_mb)

    # ---------------- pass 1：体检 + id→栈 关联表
    t0 = time.time()
    n_ev = n_stack = 0
    cats = Counter()
    names = Counter()
    durs = defaultdict(float)
    stack_by_id: dict[str, list[dict]] = {}
    stack_cats = Counter()
    sample_stack = ""
    # python 泳道的 (start, end, stack_key) 区间，供 --infer-by-time 兜底
    py_spans: list[tuple[float, float, str]] = []
    py_frames: dict[str, list[dict]] = {}
    py_capped = False

    for ev in trace.events():
        n_ev += 1
        name = str(ev.get("name") or "")
        cat = str(ev.get("cat") or "")
        cats[cat] += 1
        names[name] += 1
        try:
            durs[name] += float(ev.get("dur") or 0.0)
        except (TypeError, ValueError):
            pass
        frames = event_stack(ev)
        if frames:
            n_stack += 1
            stack_cats[cat] += 1
            if not sample_stack:
                sample_stack = "\n".join(fmt_frame(f) for f in frames[:4])
            # python 泳道 span（用于时间包含兜底）
            is_py = "python" in cat.lower() or bool(re.search(r"\.py\(\d+\)", name))
            if is_py and not py_capped:
                try:
                    st = float(ev.get("ts") or 0.0)
                    du = float(ev.get("dur") or 0.0)
                except (TypeError, ValueError):
                    st = du = 0.0
                if du > 0:
                    k = frames_key(frames)
                    py_frames.setdefault(k, frames)
                    py_spans.append((st, st + du, k))
                    if len(py_spans) >= 1_000_000:
                        py_capped = True
            eid = event_id(ev)
            if eid is not None and eid not in stack_by_id:
                stack_by_id[eid] = frames

    if args.probe:
        print("\n=== probe ===")
        print("顶层键        : %s" % (trace.top_keys or "(stream)"))
        print("事件数        : %d" % n_ev)
        print("带栈事件数    : %d (%.1f%%)" % (n_stack, 100.0 * n_stack / max(n_ev, 1)))
        print("带栈事件的 cat: %s" % dict(stack_cats.most_common(8)))
        print("所有 cat      : %s" % dict(cats.most_common(12)))
        if sample_stack:
            print("\n第一条栈（原样，前 4 帧）:\n%s" % sample_stack)
        print("\nTop 20 事件名（按总 dur，ms）:")
        for n, d in sorted(durs.items(), key=lambda kv: -kv[1])[:20]:
            print("  %10.1f ms  x%-6d %s" % (d / 1000.0, names[n], n[:80]))
        print("\nTop 12 事件名（按出现次数）:")
        for n, c in names.most_common(12):
            print("  x%-8d %s" % (c, n[:80]))
        print("\n关联 id→栈 表大小: %d" % len(stack_by_id))
        if n_stack == 0:
            print(
                "\n[!] 一个栈都没有：这次 profile 很可能没开 with_stack。\n"
                "    sglang 侧开关：managers/scheduler_components/profiler_manager.py:335 (with_stack=True)；\n"
                "    或启动参数里把 profiler 的 with_stack 打开后重采。"
            )
        print("\n用时 %.1fs" % (time.time() - t0))
        return 0

    # ---------------- 目标算子
    if args.ops_re:
        targets = [(p, re.compile(p, re.I)) for p in args.ops_re]
        mode = "--ops-re"
    elif args.top > 0:
        targets = []
        for n, _d in sorted(durs.items(), key=lambda kv: -kv[1]):
            if n and all(n != lb for lb, _ in targets):
                targets.append((n, re.compile(re.escape(n), re.I)))
            if len(targets) >= args.top:
                break
        mode = "--top"
    else:
        targets = [(lb, re.compile(pat, re.I)) for lb, pat in DEFAULT_OPS]
        mode = "default"
    # label 去重（同一 label 出现两次时加 #2）
    seen: dict[str, int] = {}
    uniq = []
    for lb, rx in targets:
        k = seen.get(lb, 0) + 1
        seen[lb] = k
        uniq.append((lb if k == 1 else "%s#%d" % (lb, k), rx))
    targets = uniq

    py_starts, py_spans = build_span_index(py_spans) if py_spans else ([], [])
    if py_spans:
        print("python span: %d 个（--infer-by-time 兜底可用）" % len(py_spans), file=sys.stderr)

    # ---------------- pass 2：抽每个算子的事件
    picked: dict[str, list[dict]] = {lb: [] for lb, _ in targets}
    for ev in trace.events():
        name = str(ev.get("name") or "")
        for lb, rx in targets:
            if not rx.search(name):
                continue
            if len(picked[lb]) >= args.per_op:
                continue
            frames = event_stack(ev)
            inferred = False
            if not frames:
                eid = event_id(ev)
                if eid is not None:
                    frames = stack_by_id.get(eid) or []
            if not frames and args.infer_by_time and py_spans:
                try:
                    ts = float(ev.get("ts") or 0.0)
                except (TypeError, ValueError):
                    ts = None
                if ts is not None:
                    k = enclosing_py_stack(py_starts, py_spans, ts)
                    if k is not None:
                        frames = py_frames.get(k) or []
                        inferred = bool(frames)
            try:
                dur = float(ev.get("dur") or 0.0)
            except (TypeError, ValueError):
                dur = 0.0
            picked[lb].append(
                {
                    "name": name,
                    "cat": str(ev.get("cat") or ""),
                    "dur": dur,
                    "frames": frames,
                    "inferred": inferred,
                    "stack": ("~" + frames_key(frames)) if (frames and inferred) else (frames_key(frames) if frames else ""),
                }
            )

    # ---------------- 渲染
    tag = args.tag or os.path.basename(os.path.abspath(args.trace).rstrip("/")) or "trace"
    out = args.out
    if out is None:
        here = os.path.abspath(os.getcwd())
        root = None
        for _ in range(6):
            if os.path.isdir(os.path.join(here, "_tmp_preflight")):
                root = here
                break
            here = os.path.dirname(here)
        base = root or os.getcwd()
        out = os.path.join(base, "_tmp_preflight", "profile_bundle", "callstacks", "%s.md" % tag)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    git_head = ""
    try:
        import subprocess

        git_head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=os.getcwd(),
        ).stdout.strip()
    except Exception:
        pass

    # 汇总（先算，用于两张表）
    rows = []
    for lb, _rx in targets:
        evs = picked[lb]
        with_st = [e for e in evs if e["frames"]]
        groups = defaultdict(list)
        for e in with_st:
            groups[e["stack"]].append(e)
        best = ""
        if groups:
            top_key = max(groups.items(), key=lambda kv: (len(kv[1]), sum(x["dur"] for x in kv[1])))[0]
            best = launcher_sglang(groups[top_key][0]["frames"]) or launcher_any(
                groups[top_key][0]["frames"]
            )
        rows.append((lb, len(evs), len(with_st), sum(e["dur"] for e in evs) / 1000.0, len(groups), best))

    L: list[str] = []
    A = L.append
    A("# Call stacks: 算子 → Python 调用栈")
    A("")
    A("> 由 `_tmp_preflight/dump_callstack.py` 生成，用于把 profiler 里的 kernel 对回 sglang 代码。")
    A("> 数据源是 `trace_view.json` 的 Python 栈事件；`kernel_details.csv` 里**没有**栈。")
    A("")
    A("| 项 | 值 |")
    A("|---|---|")
    A("| 输入 | `%s` (%s) |" % (path, _human_mb(size_mb)))
    A("| 生成时间 | %s |" % time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    A("| host / git | `%s` / `%s` |" % (socket.gethostname(), git_head or "?"))
    A(
        "| 事件数 / 带栈事件数 | %d / %d (%.1f%%) |"
        % (n_ev, n_stack, 100.0 * n_stack / max(n_ev, 1))
    )
    A("| 目标清单 | %s（%d 条） |" % (mode, len(targets)))
    A("| 关联 id→栈 表 | %d |" % len(stack_by_id))
    A(
        "| python span / 时间兜底 | %d / %s |"
        % (len(py_spans), "开（结果标 [推断]）" if args.infer_by_time and py_spans else "关（`--infer-by-time` 可开）")
    )
    A("")
    A("**读法**：")
    A("")
    A("- 每种栈按「最内层在前」打印；`launcher` = 最内层 `python/sglang/` 帧（拿不到时退回最内层任意仓库帧），")
    A("  也就是**该算子实际被哪一行 sglang 代码发起**，可直接回填 `KERNEL_CODE_MAP.md` 的「Python 发起位置」列。")
    A("- `仓库帧链` 是从内往外的完整 sglang/deep_ep 调用链（`A ⟵ B` 读作「A 被 B 调用」）。")
    A("- 栈方向按 torch 的 dispatch 帧位置自动判定；判不出来时按原始顺序打印，并在下面标注。")
    A("- 每个算子最多抽 `--per-op` 个事件、打印 `--stacks` 种栈，**是抽样不是全量**；次数列仍来自全量扫描。")
    A("")

    A("## 汇总")
    A("")
    A("| 算子 | 命中 | 带栈 | 合计 ms | 去重栈 | launcher（最内层 sglang 帧） |")
    A("|---|---|---|---|---|---|")
    for lb, n, ns, ms, ng, best in rows:
        A("| `%s` | %d | %d | %.1f | %d | %s |" % (lb, n, ns, ms, ng, ("`%s`" % best) if best else "—"))
    A("")

    zero = [lb for lb, n, *_ in rows if n == 0]
    if zero:
        A("_本次未命中：%s_" % "、".join("`%s`" % z for z in zero))
        A("")

    A("## 逐算子")
    for lb, n, ns, ms, ng, _best in rows:
        if n == 0:
            continue
        evs = picked[lb]
        A("")
        A("### `%s` — 命中 %d / 带栈 %d，合计 %.1f ms，%d 种栈" % (lb, n, ns, ms, ng))
        if not ns:
            A("")
            if n_stack == 0:
                A(
                    "> [!] 整份 trace 都没有栈：profiler 没开 `with_stack`"
                    "（sglang: `managers/scheduler_components/profiler_manager.py:335`）。"
                )
            else:
                A(
                    "> [!] 这 %d 个事件没有栈，也没有可关联（`External id`/`correlation`）的带栈事件："
                    "这类 `cat` 本身不携带栈。" % n
                )
            continue
        groups = defaultdict(list)
        for e in evs:
            if e["frames"]:
                groups[e["stack"]].append(e)
        ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), -sum(x["dur"] for x in kv[1])))
        for gi, (_key, g) in enumerate(ordered[: args.stacks], 1):
            ds = sorted(float(x["dur"]) for x in g)
            dmed = statistics.median(ds) / 1000.0
            cats = Counter(x["cat"] or "?" for x in g)
            inferred = any(x.get("inferred") for x in g)
            inner, orient = normalize_order(g[0]["frames"])
            A("")
            A(
                "**栈 #%d ×%d**  [%s]%s  dur %.1f / %.1f / %.1f ms (min/med/max)  launcher: `%s`"
                % (
                    gi,
                    len(g),
                    "/".join(c for c, _ in cats.most_common(2)),
                    " **[推断: 按时间包含]**" if inferred else "",
                    ds[0] / 1000.0,
                    dmed,
                    ds[-1] / 1000.0,
                    launcher_sglang(g[0]["frames"]) or "（栈里没有 sglang 帧）",
                )
            )
            chain = repo_chain(g[0]["frames"])
            if chain:
                A("")
                A("仓库帧链：`%s`" % chain)
            A("")
            A("```")
            for fr in inner[: args.frames]:
                A("← " + fmt_frame(fr))
            if len(inner) > args.frames:
                A("← …（共 %d 帧，省略最外层 %d 帧）" % (len(inner), len(inner) - args.frames))
            if orient != "n/a":
                A("方向: " + orient)
            A("```")
        if len(ordered) > args.stacks:
            A("")
            A("_（另有 %d 种较少见的栈未打印，`--stacks` 可调）_" % (len(ordered) - args.stacks))

    A("")
    A("## 回填用速查（可直接贴进 KERNEL_CODE_MAP.md 的映射表）")
    A("")
    A("| 算子 | 本次 trace 次数 | launcher（Python 发起位置） |")
    A("|---|---|---|")
    for lb, n, ns, ms, ng, best in rows:
        A("| `%s` | %d | %s |" % (lb, n, ("`%s`" % best) if best else "（无栈，未验证）"))
    A("")

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    print(
        "wrote %s (%s, %d lines)"
        % (out, _human_mb(os.path.getsize(out) / 1024 / 1024), len(L))
    )
    if n_stack == 0:
        print("[!] 整份 trace 没有任何栈事件 —— 先跑 --probe 看原因。", file=sys.stderr)
    print("用时 %.1fs" % (time.time() - t0), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
