#!/usr/bin/env python3
"""analyze_prefill_steps.py —— 把 128K prefill 的 trace 拆成「每 chunk 一行」，回答两件事：

  1. 一次 128K/50% prefill 里，时间花在**扫 prefix 的 attention**、**新 token 的 FFN/MoE**、
     还是**海量小算子/通信**上？（三选一决定 2.46× 押哪条路）
  2. attention 的每 chunk 耗时**是否随已扫 prefix 长度增长**？（§5.5 只能把份额夹在 35%~84%，
     这条斜率就是把它定死的那把尺子）

为什么不用 step_trace_time.csv：实测它只有**一行**汇总（Step 列为空），没有每步数据；
所以这里从 kernel_details.csv 的 Start Time 自己切步。

切步策略（逐级降级，全部失败就老实报错，不会瞎猜）：
  A. 有 Model ID 且取值 >1 种 → 直接用它；
  B. 否则用主计算流 kernel 名序列的**周期性**：FFT 自相关找周期 L，再用精确匹配验证
     （同一 chunk 内 kernel 序列完全重复，decode 段会破坏周期性 → 顺带定出 prefill 步数）；
  C. 都不行 → 打印诊断，只出总量表。

用法：
  python3 _tmp_preflight/analyze_prefill_steps.py /tmp/prof_128k
  # 冷启动 trace（prefix 4K→131K，32 chunk）：工作点窗口按 T_k>=65536 取 16 个 chunk
  python3 _tmp_preflight/analyze_prefill_steps.py /tmp/prof_128k_cold --chunk-tokens 4096 \\
      --cached-prefix 0 --total-tokens 129093 --window-from 65536 --csv steps.csv

`--cached-prefix`（本次命中进 cache 的前缀）与 `--total-tokens`（未命中的 token 数）
只影响 x 轴的绝对位置和「prefix 部分」的归因；斜率（ms/1K token）与两者无关。
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_profile import bucket_of  # noqa: E402  复用同一套分桶规则

BUCKET_ORDER = ["attention", "matmul/linear", "MoE/gating", "rope/norm", "elementwise",
                "HCCL/通信", "其它"]

# 回传块里的 key 只用 ASCII，别让中文进 git/表格
LABEL = {"attention": "ATTN", "matmul/linear": "MM", "MoE/gating": "MOE", "rope/norm": "ROPE",
         "elementwise": "ELEM", "HCCL/通信": "COMM", "其它": "OTHER"}

CMAP = {
    "name": ("Name",),
    "stream": ("Stream ID",),
    "start": ("Start Time(us)", "Start Time(us)"),
    "dur": ("Duration(us)",),
    "core": ("Accelerator Core",),
    "model": ("Model ID",),
    "aicore": ("aicore_time(us)",),
}


# ---------------------------------------------------------------- 读 trace
def _dir_weight(d: str) -> int:
    kd = os.path.join(d, "ASCEND_PROFILER_OUTPUT", "kernel_details.csv")
    try:
        return os.path.getsize(kd)
    except OSError:
        return 0


def find_rank_dir(root: str, want: str | None) -> str:
    cands = sorted(glob.glob(os.path.join(root, "*_ascend_pt")))
    if not cands:
        if os.path.isdir(os.path.join(root, "ASCEND_PROFILER_OUTPUT")):
            return root
        sys.exit("没找到 *_ascend_pt 目录：%s" % root)
    if want:
        hit = [d for d in cands if want in os.path.basename(d)]
        if not hit:
            sys.exit("--rank %r 没匹配到任何目录；候选：%s" % (want, [os.path.basename(d) for d in cands]))
        return hit[0]
    ranked = sorted(cands, key=_dir_weight, reverse=True)
    if len(ranked) > 1:
        print("发现 %d 个 rank 目录，取最有料的一个（其余是空闲 replica）：" % len(ranked))
        for d in ranked:
            print("    %9.1f MB  %s" % (_dir_weight(d) / 1e6, os.path.basename(d)))
    return ranked[0]


def _col_index(header: list[str]) -> dict:
    """按需找列，容忍前后空格/大小写。"""
    norm = {(h or "").strip(): i for i, h in enumerate(header)}
    out = {}
    for key, names in CMAP.items():
        for n in names:
            if n in norm:
                out[key] = norm[n]
                break
    return out


def read_kernels(path: str) -> tuple[list[dict], dict]:
    """只读需要的列，避免几百 MB 的 DictReader 开销。"""
    if not os.path.isfile(path):
        sys.exit("没有 kernel_details.csv：%s" % path)
    rows = []
    with open(path, newline="", errors="replace") as f:
        rd = csv.reader(f)
        try:
            header = next(rd)
        except StopIteration:
            sys.exit("kernel_details.csv 是空的：%s" % path)
        idx = _col_index(header)
        missing = [k for k in ("name", "stream", "start", "dur") if k not in idx]
        if missing:
            sys.exit("kernel_details.csv 缺列 %s；实际表头：%s" % (missing, header[:20]))
        skip_nodur = 0
        for r in rd:
            try:
                dur = float(r[idx["dur"]])
            except (ValueError, IndexError):
                skip_nodur += 1
                continue
            if dur <= 0:
                skip_nodur += 1
                continue
            try:
                start = float(r[idx["start"]])
            except (ValueError, IndexError):
                skip_nodur += 1
                continue
            rows.append({
                "name": (r[idx["name"]] or "").strip(),
                "stream": (r[idx["stream"]] or "?").strip(),
                "start": start,
                "dur": dur,
                "core": (r[idx["core"]] or "?").strip() if "core" in idx else "?",
                "model": (r[idx["model"]] or "").strip() if "model" in idx else "",
                "aicore": None,
            })
            if "aicore" in idx:
                try:
                    rows[-1]["aicore"] = float(r[idx["aicore"]])
                except (ValueError, IndexError):
                    pass
    return rows, {"header": header, "skipped": skip_nodur, "cols": idx}


# ---------------------------------------------------------------- 找主计算流
def pick_main_stream(rows: list[dict]) -> tuple[str, list[dict]]:
    """主计算流 = **AI 核上耗时最多**的流。

    smoke 实测（prof_smoke3）踩过两个坑：
      · 按「kernel 数最多」选 → 选中 stream 76（29,632 个微秒级 vector kernel，仅 127ms），
        真正的主力是 stream 43（15,725 个 kernel，1239ms）；
      · 按「总耗时最多」选而不排除 AI_CPU → stream 96 的 allreduceAicpuKernel 有 1187ms，
        几乎追平 43，但它是通信调度，不是计算。
    所以只统计 core 不是 AI_CPU / COMMUNICATION 的 kernel 时长。
    """
    per = defaultdict(lambda: [0, 0.0])
    for r in rows:
        d = per[r["stream"]]
        d[0] += 1
        core = (r.get("core") or "").upper()
        if "CPU" not in core and "COMM" not in core:
            d[1] += r["dur"]
    cands = [(sid, n, us) for sid, (n, us) in per.items() if sid != "N/A"]
    if not cands:
        cands = [(sid, n, us) for sid, (n, us) in per.items()]
    cands.sort(key=lambda kv: -kv[2])
    print("按 AI 核耗时排前 5 的流（count 仅参考）：")
    for sid, n, us in cands[:5]:
        print("    stream %-6s kernels=%-8d AI核耗时=%.1f ms" % (sid, n, us / 1000.0))
    sid = cands[0][0]
    return sid, sorted([r for r in rows if r["stream"] == sid], key=lambda r: r["start"])


# ---------------------------------------------------------------- 切步
def _encode(names: list[str]) -> list[int]:
    vocab, ids = {}, []
    for n in names:
        v = vocab.get(n)
        if v is None:
            v = len(vocab)
            vocab[n] = v
        ids.append(v)
    return ids


def detect_period(ids: list[int]) -> tuple[int | None, int, str]:
    """FFT 自相关找周期 L；返回 (L, 有效周期前缀长度, 诊断)。"""
    try:
        import numpy as np
    except ImportError:
        return None, 0, "没有 numpy，无法做周期检测"
    N = len(ids)
    if N < 8:
        return None, 0, "kernel 数太少（%d）" % N
    x = np.asarray(ids, dtype=np.float64)
    x -= x.mean()
    n = 1 << int(np.ceil(np.log2(2 * N)))
    f = np.fft.rfft(x, n)
    ac = np.fft.irfft(f * np.conj(f), n)[:N]
    if ac[0] <= 0:
        return None, 0, "自相关退化（全同 kernel？）"
    ac = ac / ac[0]

    lo, hi = max(8, N // 64), min(N - 1, max(16, N // 3))
    if hi <= lo:
        return None, 0, "可搜索的周期范围为空（N=%d）" % N
    cand = sorted(range(lo, hi), key=lambda L: -ac[L])[:24]

    ids_arr = np.asarray(ids)
    for L in sorted(cand):
        m = ids_arr[L:] == ids_arr[:-L]
        score = float(m.mean())
        if score < 0.85:
            continue
        # 周期性在哪结束？用 32 窗的匹配率<90% 定位（decode 段会破坏周期）
        w = min(32, max(1, len(m) // 4))
        cum = np.concatenate([[0], np.cumsum(m.astype(np.int64))])
        if len(m) >= w:
            means = (cum[w:] - cum[:-w]) / float(w)
            bad = np.flatnonzero(means < 0.9)
            end = int(bad[0] + w) if len(bad) else len(m)
        else:
            end = len(m)
        Lgood = end + L
        n_steps = int(round(Lgood / L))
        if n_steps < 2:
            continue
        return L, Lgood, "周期=%d，整体匹配=%.3f，周期区=%d，推得步数=%d" % (
            L, score, Lgood, n_steps)
    return None, 0, "24 个候选周期都没有通过精确匹配（匹配率<0.85），可能 kernel 序列不严格周期"


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def segment_by_attention_duration(rows, main, tol=0.015):
    """按 attention kernel 的 duration 阶跃切 chunk（主方法）。

    物理依据：一个 chunk 内，**所有 full-attention 层**看到的 seq_len 相同（都是 T_k），
    所以同一个算子名的 duration 在 chunk 内基本恒定；到下一个 chunk，T_k 增加 4096，
    duration 随之阶跃。于是「连续的等值段」= 一个 chunk。

    比 kernel 名序列的周期性可靠得多：真实模型里每层的算子序列往往**完全相同**，
    按名字找周期会锁到「一层的周期」上，把步数放大 num_layers 倍。
    SWA 层的 kernel 会饱和成常量（只形成 1 段），所以自然不会被选中。

    返回 (bounds, n_steps, 诊断串) 或 (None, 0, 原因)。
    """
    att = [r for r in rows if bucket_of(r["name"]) == "attention"]
    if len(att) < 4:
        return None, 0, "attention kernel 太少（%d）" % len(att)
    by_name = {}
    for r in att:
        by_name.setdefault(r["name"], []).append(r)

    best = None
    for nm, lst in by_name.items():
        if len(lst) < 2:
            continue
        lst.sort(key=lambda r: r["start"])
        groups = [[lst[0]]]
        for r in lst[1:]:
            cur = _median([x["dur"] for x in groups[-1]])
            if cur > 0 and abs(r["dur"] - cur) <= tol * cur:
                groups[-1].append(r)
            else:
                groups.append([r])
        if len(groups) < 2:
            continue
        sizes = [len(g) for g in groups]
        imbal = (max(sizes) - min(sizes)) / float(max(sizes))
        if imbal > 0.25:            # 每组 kernel 数应基本一致（同一批层）
            continue
        # 偏好：组数多、imbalance 小、名字出现次数多
        score = (len(groups), -imbal, len(lst))
        if best is None or score > best[0]:
            best = (score, nm, groups, imbal)

    if best is None:
        return None, 0, "没有哪个 attention 算子名的 duration 能切成 ≥2 个等长段"

    _score, nm, groups, imbal = best
    starts = [g[0]["start"] for g in groups]
    main_starts = [r["start"] for r in main]
    import bisect
    bounds = []
    for t0 in starts:
        i = bisect.bisect_left(main_starts, t0)
        bounds.append(min(i, len(main) - 1))
    bounds = sorted(set(bounds))
    n_steps = len(bounds)
    why = ("duration 阶跃切步：算子=%s，段数=%d，每组 kernel 数≈%d，imbalance=%.2f"
           % (nm[:40], n_steps, len(groups[0]), imbal))
    return bounds, n_steps, why


def _snap_back(main, idx: int, thr_us: float, lookback: int = 500) -> int:
    """把一个下标往前挪到「它所在的那段连续 kernel」的开头。

    chunk 边界处有明显空档（smoke 实测：chunk 内相邻 kernel 间隔 ~10ms，
    跨 chunk ~512ms），所以从锚点往回走到第一个大空档即可。
    """
    j = idx
    lo = max(0, idx - lookback)
    while j > lo:
        gap = main[j]["start"] - (main[j - 1]["start"] + main[j - 1]["dur"])
        if gap >= thr_us:
            break
        j -= 1
    return j


def segment_by_anchor(main, expected, max_count=256, ratio_lo=0.5, ratio_hi=2.0):
    """用一个「每个 forward 恰好出现 N 次」的算子做锚点切步（最稳，smoke 定案）。

    为什么要它：真实 trace 里 SparseFlashMla 每个 forward 出现 42 次、duration 从 20us
    到 3728us（SWA 层饱和、全注意力层随 prefix 涨），**同一 forward 内就跨两个数量级**，
    所以无论是「按名字找周期」还是「按 duration 阶跃」都切不出来。
    而 `_expand_index_page_table_kernel` 这类算子每个 forward 只出现一次
    （smoke 实测 count=3 / 3 forward），用它当 chunk 标记最干净。
    """
    if expected and expected >= 2:
        pass
    else:
        return None, 0, "没有期望步数，跳过锚点法"
    by_name = {}
    for i, r in enumerate(main):
        by_name.setdefault(r["name"], []).append(i)

    # 窗口总跨度：一个合格的「每 forward 一次」的锚点必须**铺满整个窗口**。
    # 只按「次数对 + 间隔均匀」会选到挤在 1ms 内的 3 次调用（smoke 实踩：
    # aclnnRemainderTensorTensor_FloorModA count=3，间隔 CV=0.10，但三次都在同一毫秒），
    # 结果切出 3 个 0.2ms 的"步"。
    span = main[-1]["start"] - main[0]["start"]
    best = None
    rejected = []
    for nm, idxs in by_name.items():
        c = len(idxs)
        if c < 2 or c > max_count:
            continue
        starts = [main[i]["start"] for i in idxs]
        gaps = [b - a for a, b in zip(starts, starts[1:])]
        if not gaps:
            continue
        mean = sum(gaps) / len(gaps)
        if mean <= 0:
            continue
        # 均匀度：间隔离散程度
        cv = (sum((g - mean) ** 2 for g in gaps) / len(gaps)) ** 0.5 / mean
        if cv > 0.6:
            continue
        # 铺满度：平均间隔应当接近 跨度/次数（≈1 表示均匀铺满整个窗口）
        ratio = mean / (span / c) if span > 0 else 0.0
        if not (ratio_lo <= ratio <= ratio_hi):
            rejected.append((nm, c, ratio))
            continue
        per = c // expected if expected and c % expected == 0 else None
        misfit = 0 if per else abs(c - expected) if expected else 0
        # 偏好：每 forward 恰好 1 次（per=1）、间隔最均匀
        score = (misfit, 0 if per == 1 else 1, cv, -c)
        if best is None or score < best[0]:
            best = (score, nm, idxs, per, cv, c, ratio)
    if best is None:
        msg = "主计算流上找不到「每 forward 固定次数」的锚点算子"
        if rejected:
            msg += "（%d 个候选因没铺满窗口被否，例如 %s count=%d ratio=%.3f）" % (
                len(rejected), rejected[0][0][:30], rejected[0][1], rejected[0][2])
        return None, 0, msg

    _score, nm, idxs, per, cv, c, ratio = best
    if per and per > 1:
        idxs = idxs[::per]

    # 空档阈值：主计算流相邻 kernel 间隔的中位数 × 20（smoke：~10ms ⇒ 阈值 200ms）
    gaps = [main[i]["start"] - (main[i - 1]["start"] + main[i - 1]["dur"])
            for i in range(1, len(main))]
    pos = sorted(g for g in gaps if g > 0)
    med = pos[len(pos) // 2] if pos else 0.0
    thr = max(20_000.0, 10 * med)          # us
    # 回看窗口最多半个 chunk，避免越过上一个 chunk 的边界
    lookback = max(20, int(len(main) / max(len(idxs), 1) / 2))
    snapped = sorted({_snap_back(main, i, thr, lookback) for i in idxs})
    if len(snapped) != len(idxs):          # 吸附把边界合并了 ⇒ 放弃吸附，用原始锚点
        bounds = sorted(set(idxs))
        why_snap = "（吸附合并了边界，已放弃吸附）"
    else:
        bounds = snapped
        why_snap = ""
    why = ("锚点=%s count=%d 每forward=%s 间隔CV=%.2f 铺满度=%.2f 空档阈值=%.0fms → %d 步%s"
           % (nm[:36], c, per or "?", cv, ratio, thr / 1000.0, len(bounds), why_snap))
    return bounds, len(bounds), why


def segment_by_model(main: list[dict]) -> int:
    """用 Model ID 直接当步号；返回步数（0=不可用）。"""
    ids = [r["model"] for r in main if r["model"] not in ("", "N/A")]
    if len(ids) < len(main) * 0.8:
        return 0
    uniq = sorted(set(ids))
    if len(uniq) < 2:
        return 0
    return len(uniq)


def build_steps(main: list[dict], L: int, Lgood: int) -> tuple[list[int], int]:
    """返回 (每步在主计算流里的起始下标列表, 步数)。"""
    n_steps = max(1, int(round(Lgood / L)))
    bounds = [k * L for k in range(n_steps)]
    return bounds, n_steps


# ---------------------------------------------------------------- 统计
def summarise(rows, main, bounds, n_steps):
    """每步：wall + 各桶耗时。主计算流按下标切，其它流按时间窗切。

    最后一步的右边界怎么定：每个 chunk 的 kernel 数应当一致，所以取
    ``bounds[-1] + (bounds[1]-bounds[0])`` 处的 kernel 起点作为 prefill 的终点；
    这样 decode 段不会被算进最后一个（也是最贵的）chunk。
    """
    t = [main[i]["start"] for i in bounds]
    per_chunk = (bounds[1] - bounds[0]) if n_steps >= 2 else 0
    tail_idx = bounds[-1] + per_chunk
    tail_start = main[tail_idx]["start"] if per_chunk and tail_idx < len(main) else None
    t_end = tail_start if tail_start is not None else max(r["start"] + r["dur"] for r in main)
    edges = t + [t_end]

    steps = [{
        "wall": 0.0, "n": 0, "pk": defaultdict(float), "aicore": 0.0, "model": "",
    } for _ in range(n_steps)]
    for k in range(n_steps):
        steps[k]["wall"] = edges[k + 1] - edges[k]

    import bisect
    for r in rows:
        k = bisect.bisect_right(edges, r["start"]) - 1
        if k < 0:
            k = 0
        if k >= n_steps:
            continue  # decode/尾部
        s = steps[k]
        s["n"] += 1
        s["pk"][bucket_of(r["name"])] += r["dur"]
        if r["aicore"]:
            s["aicore"] += r["aicore"]
    return steps, edges


def fit_slope(xs, ys):
    """一次线性回归，返回 (slope_us_per_token, intercept_us, r2)。"""
    try:
        import numpy as np
    except ImportError:
        return None, None, None
    n = len(xs)
    if n < 3:
        return None, None, None
    X = np.asarray(xs, dtype=np.float64)
    Y = np.asarray(ys, dtype=np.float64)
    A = np.vstack([X, np.ones(n)]).T
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    pred = A @ coef
    ss_res = float(((Y - pred) ** 2).sum())
    ss_tot = float(((Y - Y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(coef[0]), float(coef[1]), r2


def window_stats(steps, cum, lo: int) -> dict:
    """工作点窗口：只统计 T_k>=lo 的 chunk，返回组成份额与斜率归因。"""
    idx = [k for k in range(len(steps)) if cum[k] >= lo]
    if not idx:
        return {}
    wall = sum(steps[k]["wall"] for k in idx) or 1.0
    tot = {b: sum(steps[k]["pk"].get(b, 0.0) for k in idx) for b in BUCKET_ORDER}
    xs = [cum[k] for k in idx]
    ys = [steps[k]["pk"].get("attention", 0.0) for k in idx]
    slope, _icept, r2 = fit_slope(xs, ys)
    f_prefix = (slope * sum(xs) / wall) if slope is not None else 0.0
    return {
        "n": len(idx), "wall": wall, "tot": tot, "t0": cum[idx[0]], "t1": cum[idx[-1]],
        "f": {b: tot[b] / wall for b in BUCKET_ORDER},
        "f_prefix": f_prefix, "slope": slope, "r2": r2,
    }


def fmt_ms(us: float) -> str:
    return "%.1f" % (us / 1000.0)


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--rank", default=None, help="只分析目录名含该子串的 rank")
    ap.add_argument("--chunk-tokens", type=int, default=4096, help="每个 prefill chunk 的 token 数")
    ap.add_argument("--cached-prefix", type=int, default=0, help="第 0 个 chunk 之前已命中缓存的前缀长度")
    ap.add_argument("--total-tokens", type=int, default=0, help="未命中的总 token 数（0=按步数×chunk 估）")
    ap.add_argument("--csv", default=None, help="把每步明细写成 CSV（体积很小，可进证据包）")
    ap.add_argument("--seg-tol", type=float, default=0.015,
                    help="attention duration 分段容差（默认 1.5%%；chunk 变大时每步约 +3%%）")
    ap.add_argument("--window-from", type=int, default=0,
                    help="再报一个「工作点窗口」的组成：只统计累计长度 T_k>=该值的 chunk"
                         "（冷启动 trace 里用 65536 就正好是 50%% 命中的那 16 个 chunk）")
    ap.add_argument("--ttft-ms", type=float, default=73710.0,
                    help="bench 实测的 TTFT（用来把份额换算成秒；默认 73.71s = 新树基线）")
    ap.add_argument("--target-ms", type=float, default=30000.0, help="TTFT 目标（默认 30s）")
    args = ap.parse_args()

    rd = find_rank_dir(args.root, args.rank)
    ap_dir = os.path.join(rd, "ASCEND_PROFILER_OUTPUT")
    print("分析：%s" % ap_dir)
    rows, meta = read_kernels(os.path.join(ap_dir, "kernel_details.csv"))
    if not rows:
        sys.exit("kernel_details.csv 里没有可用行")
    print("kernel 行数=%d（跳过 dur<=0 的 %d 行）" % (len(rows), meta["skipped"]))

    main_sid, main_rows = pick_main_stream(rows)
    print("主计算流 = %s（%d 个 kernel）" % (main_sid, len(main_rows)))

    # ---- 切步（按可靠性排序：物理信号 > 框架元数据 > 序列周期性）----
    bounds = None
    n_steps = 0
    method = ""

    # 期望 prefill 步数：由 token 数与 chunk 数算出（smoke/full 都适用）
    expected = -(-args.total_tokens // args.chunk_tokens) if (
        args.total_tokens and args.chunk_tokens) else 0

    # 1) 锚点算子（每 forward 固定出现次数）——smoke 实测最稳
    b0, n0, why0 = segment_by_anchor(main_rows, expected)
    if not b0:
        # 严格阈值（铺满度 0.5~2）失败时，放宽到 0.15~6 再试一次，避免又多一轮往返
        b0, n0, why0b = segment_by_anchor(main_rows, expected,
                                         ratio_lo=0.15, ratio_hi=6.0)
        if b0:
            why0 = why0b + "（已放宽铺满度阈值）"
            print("  [warn] 严格阈值下没有锚点，已放宽重试")
    print("切步[1/3] 锚点算子：%s" % why0)
    if b0 and n0 >= 2 and (not expected or abs(n0 - expected) <= 2):
        bounds, n_steps, method = b0, n0, "anchor"
    elif True:
        b1, n1, why1 = segment_by_attention_duration(rows, main_rows, tol=args.seg_tol)
        print("切步[2/3] attention-duration：%s" % why1)
        if b1 and n1 >= 2 and (not expected or abs(n1 - expected) <= 2):
            bounds, n_steps, method = b1, n1, "attention-duration"
        else:
            # 3) Model ID（smoke 实测：只有 3 个取值，其中 4294967295=0xFFFFFFFF 是哨兵，不可用）
            n_model = segment_by_model(main_rows)
            if n_model >= 2:
                groups = defaultdict(list)
                for i, r in enumerate(main_rows):
                    groups[r["model"]].append(i)
                bounds = [min(v) for _k, v in sorted(groups.items())]
                n_steps = len(bounds)
                method = "Model ID"
                print("切步[3/3] Model ID：%d 步" % n_steps)
            else:
                # 4) kernel 名序列的周期性（真实模型里每层算子序列通常相同 → 会锁到「一层」
                #    的周期，把步数放大 num_layers 倍，所以只当最后退路）
                ids = _encode([r["name"] for r in main_rows])
                L, Lgood, why4 = detect_period(ids)
                print("切步[3/3] Model ID：不可用（smoke 实测只有 3 个取值，含 0xFFFFFFFF 哨兵）")
                print("切步[4/4] 周期检测：%s" % why4)
                if not L:
                    print("\n[FAIL] 四种切步法都失败，只能出总量表（见下）。")
                    return _aggregate_only(rows, main_sid, args)
                bounds, n_steps = build_steps(main_rows, L, Lgood)
                method = "周期检测(L=%d)" % L
                print("  [warn] 退路方法：若步数≈num_layers×chunk数，说明锁到了「一层」的周期")

    print("切步：%s → %d 步（第 0 步起点 t=%.3f ms）" % (
        method, n_steps, main_rows[bounds[0]]["start"] / 1000.0))

    # 与理论步数对账：129,093 / 4096 ≈ 32
    if args.total_tokens and args.chunk_tokens:
        expect = -(-args.total_tokens // args.chunk_tokens)
        if abs(n_steps - expect) > 2:
            print("  [warn] 切出 %d 步，但按 %d token / %d = %d 步。差得远就别信下面的表"
                  % (n_steps, args.total_tokens, args.chunk_tokens, expect))

    steps, edges = summarise(rows, main_rows, bounds, n_steps)

    # ---- 每个 chunk 的 token 数与累计长度 ----
    chunk = args.chunk_tokens
    total = args.total_tokens or chunk * n_steps
    toks, cum, left = [], [], total
    for k in range(n_steps):
        tk = min(chunk, left) if left > 0 else 0
        toks.append(tk)
        left -= tk
        cum.append(args.cached_prefix + sum(toks))
    if args.total_tokens and total > chunk * n_steps:
        print("[warn] total_tokens=%d > chunk*n_steps=%d：步数可能被截断（NUM_STEPS 太小？）"
              % (total, chunk * n_steps))

    # ---- 每步表 ----
    print("\n===== 每 chunk 明细（wall=墙钟，其余=该桶 kernel 耗时之和） =====")
    head = "step %9s %7s %9s" % ("T_k(tok)", "tokens", "wall_ms")
    for b in BUCKET_ORDER:
        head += " %10s" % b.split("/")[0][:9]
    head += " %7s" % "aicore%"
    print(head)
    for k, s in enumerate(steps):
        line = "%-4d %9d %7d %9s" % (k, cum[k], toks[k], fmt_ms(s["wall"]))
        for b in BUCKET_ORDER:
            line += " %10s" % fmt_ms(s["pk"].get(b, 0.0))
        busy = (100.0 * s["aicore"] / s["wall"]) if s["wall"] > 0 and s["aicore"] else float("nan")
        line += " %7s" % ("%.0f" % busy if busy == busy else "-")
        print(line)

    walls = sorted(s["wall"] for s in steps)
    med = walls[len(walls) // 2] if walls else 0.0
    # 只有「首步明显比中位步大」才是相位/拼接问题；早期 chunk 天生更便宜（prefix 短），
    # 所以「首步更小」是正常的，不要告警。
    if steps and med > 0 and steps[0]["wall"] > 1.3 * med:
        print("  [warn] 第 0 步墙钟 %.1f ms > 中位步 %.1f ms 的 1.3 倍：可能有 prologue 混进来了"
              % (steps[0]["wall"] / 1000.0, med / 1000.0))

    prefill_wall = sum(s["wall"] for s in steps)
    print("\nprefill 墙钟合计 = %.1f ms（%d 步；主计算流 t0=%.0f ms，末尾=%.0f ms）"
          % (prefill_wall / 1000.0, n_steps, edges[0] / 1000.0, edges[-1] / 1000.0))

    # ---- 各桶总量与斜率 ----
    print("\n===== 各桶：总量 / 相对墙钟 / 随累计长度的斜率 =====")
    print("%-14s %10s %8s %14s %10s %8s" % ("桶", "总耗时ms", "占墙钟", "斜率 ms/1Ktok", "截距ms", "R²"))
    fitted = {}
    for b in BUCKET_ORDER:
        ys = [s["pk"].get(b, 0.0) for s in steps]
        tot = sum(ys)
        if tot <= 0:
            continue
        slope, icept, r2 = fit_slope(cum, ys)
        fitted[b] = (slope, icept, r2, tot)
        print("%-14s %10s %7.1f%% %14s %10s %8s" % (
            b, fmt_ms(tot), 100.0 * tot / prefill_wall if prefill_wall else 0,
            ("%.3f" % slope) if slope is not None else "-",   # us/token 数值上等于 ms/1K token
            ("%.1f" % (icept / 1000.0)) if icept is not None else "-",
            ("%.3f" % r2) if r2 is not None and r2 == r2 else "-"))

    def slope_ms_per_1k(b):
        """us/token → ms/1K token：×1000/1000 = 原值（数值相等，单位不同）。"""
        v = fitted.get(b)
        return None if not v or v[0] is None else v[0]

    # ---- 结论块 ----
    def share(b):
        s = fitted.get(b)
        return (s[3] / prefill_wall) if s and prefill_wall else 0.0

    attn = fitted.get("attention")
    f_attn = share("attention")
    f_prefix = 0.0
    if attn and attn[0] is not None:
        f_prefix = (attn[0] * sum(cum)) / prefill_wall if prefill_wall else 0.0
    small = share("elementwise") + share("rope/norm")
    comm = share("HCCL/通信")
    moe = share("MoE/gating") + share("matmul/linear")

    print("\n===== 归因（prefill 墙钟 = %.1f s） =====" % (prefill_wall / 1e6))
    print("  attention 全部            : %6.1f%%   (斜率 %s ms/1K token)" % (
        100 * f_attn, "%.3f" % slope_ms_per_1k("attention") if slope_ms_per_1k("attention") is not None else "-"))
    print("  其中「随 prefix 线性增长」: %6.1f%%   ← 这部分才是稀疏/压缩 KV、CP 能吃掉的" % (100 * f_prefix))
    print("  小算子（elem+rope/norm）  : %6.1f%%" % (100 * small))
    print("  通信                      : %6.1f%%" % (100 * comm))
    print("  FFN/MoE（moe+matmul）     : %6.1f%%" % (100 * moe))
    if f_prefix > 0:
        print("\n  若 prefix 扫描免费：上限 %.2f×（1/(1-f_prefix)）" % (1.0 / (1.0 - f_prefix)))
    need_ms = args.ttft_ms - args.target_ms
    print("\n  TTFT 假设 = %.1f s，目标 %.1f s → 需砍 %.1f s" % (
        args.ttft_ms / 1000.0, args.target_ms / 1000.0, need_ms / 1000.0))
    print("  （份额来自本 trace；把份额乘到 bench 的 TTFT 上，才可比）")
    for label, f in (("砍光 prefix 扫描", f_prefix), ("砍光 attention(含新token)", f_attn),
                     ("砍光小算子", small), ("砍光通信", comm), ("砍光 FFN/MoE+matmul", moe)):
        save = f * args.ttft_ms
        print("    · %-24s → 省 %6.1f s %s" % (label, save / 1000.0,
                                             "✅ 够了" if save >= need_ms else "（不够）"))

    # ---- 工作点窗口（冷启动 trace 推导「50% 命中时的那 16 个 chunk」）----
    win = {}
    if args.window_from > 0:
        win = window_stats(steps, cum, args.window_from)
        if not win:
            print("\n[warn] --window-from %d 没有选中任何 chunk（T_max=%d）"
                  % (args.window_from, cum[-1]))
        else:
            print("\n===== 工作点窗口：T_k >= %d（%d 个 chunk，T %d→%d）=====" % (
                args.window_from, win["n"], win["t0"], win["t1"]))
            print("  窗口墙钟合计 = %.1f s   ← 与 bench 的 TTFT 对照（50%% 命中基线 73.71s）"
                  % (win["wall"] / 1e6))
            print("%-14s %10s %8s" % ("桶", "总耗时ms", "占窗口"))
            for b in BUCKET_ORDER:
                if win["tot"][b] > 0:
                    print("%-14s %10s %7.1f%%" % (b, fmt_ms(win["tot"][b]), 100 * win["f"][b]))
            print("  attention 随 prefix 增长部分 = %.1f%%（斜率 %s ms/1K token，R²=%s）" % (
                100 * win["f_prefix"],
                "%.3f" % win["slope"] if win["slope"] is not None else "-",
                ("%.3f" % win["r2"]) if win["r2"] is not None and win["r2"] == win["r2"] else "-"))
            print("  ⇒ 用这一列去判 2.46× 押哪条路（不要用全 trace 的列，它混了冷/热两段）")

    # ---- 回传块 ----
    print("\n===== 回传块（复制这段）=====")
    print("STEPS=%d METHOD=%s CHUNK=%d CACHED_PREFIX=%d TOTAL=%d" % (
        n_steps, method, chunk, args.cached_prefix, total))
    print("PREFILL_WALL_S=%.2f" % (prefill_wall / 1e6))
    print("F_ATTN=%.3f F_PREFIX=%.3f F_SMALL=%.3f F_COMM=%.3f F_MOE_MM=%.3f" % (
        f_attn, f_prefix, small, comm, moe))
    for b in ("attention", "matmul/linear", "MoE/gating", "elementwise", "HCCL/通信"):
        v = fitted.get(b)
        if v and v[0] is not None:
            print("SLOPE_%s=%.3f_ms_per_1K R2=%s TOTAL_S=%.2f" % (
                LABEL[b], v[0], ("%.3f" % v[2]) if v[2] == v[2] else "-", v[3] / 1e6))
    if win:
        print("WIN_FROM=%d WIN_N=%d WIN_T=%d..%d WIN_WALL_S=%.2f" % (
            args.window_from, win["n"], win["t0"], win["t1"], win["wall"] / 1e6))
        print("WIN_F_ATTN=%.3f WIN_F_PREFIX=%.3f WIN_F_SMALL=%.3f WIN_F_COMM=%.3f WIN_F_MOE_MM=%.3f"
              % (win["f"]["attention"], win["f_prefix"],
                 win["f"]["elementwise"] + win["f"]["rope/norm"], win["f"]["HCCL/通信"],
                 win["f"]["MoE/gating"] + win["f"]["matmul/linear"]))
        print("WIN_SLOPE_ATTN=%s_ms_per_1K R2=%s" % (
            "%.3f" % win["slope"] if win["slope"] is not None else "-",
            ("%.3f" % win["r2"]) if win["r2"] is not None and win["r2"] == win["r2"] else "-"))
    print("===== 回传块结束 =====")

    # ---- CSV ----
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "T_k_tokens", "tokens", "wall_ms"] + [b for b in BUCKET_ORDER] + ["aicore_pct"])
            for k, s in enumerate(steps):
                busy = (100.0 * s["aicore"] / s["wall"]) if s["wall"] > 0 and s["aicore"] else ""
                w.writerow([k, cum[k], toks[k], "%.3f" % (s["wall"] / 1000.0)]
                           + ["%.3f" % (s["pk"].get(b, 0.0) / 1000.0) for b in BUCKET_ORDER]
                           + [("%.1f" % busy) if busy != "" else ""])
        print("\n每步明细已写：%s" % args.csv)
    return 0


def _aggregate_only(rows, main_sid, args) -> int:
    """切步失败时的兜底：只出总量桶表，明确标注不足以定斜率。"""
    total = sum(r["dur"] for r in rows) or 1.0
    agg = defaultdict(float)
    for r in rows:
        agg[bucket_of(r["name"])] += r["dur"]
    print("\n===== 兜底：全 trace 桶分布（不含每步斜率）=====")
    for b, us in sorted(agg.items(), key=lambda kv: -kv[1]):
        print("  %-14s %10s %7.1f%%" % (b, fmt_ms(us), 100.0 * us / total))
    print("\n注意：这是所有 stream 相加，> 墙钟；且**无法**回答「attention 是否随 prefix 增长」。")
    print("建议：加 --rank 指定具体 rank，或先看上面的切步诊断再决定补采。")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
