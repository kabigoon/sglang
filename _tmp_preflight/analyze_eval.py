#!/usr/bin/env python3
"""分析 evalscope 的评测产物：finish_reason 分布、答案完整性、得分情况。

用法（在当初执行 eval_gsm8k.sh 的目录下）：
    python3 _tmp_preflight/analyze_eval.py                # 自动找最新的 evalscope_out_*
    python3 _tmp_preflight/analyze_eval.py <work_dir>      # 指定目录

解决两个坑：
  1. finish_reason 不一定在记录顶层（evalscope 1.15 存在嵌套里）→ 递归搜索；
  2. 字段名随版本变化 → 先打印真实 keys，再按实际结构取值。
"""
import collections
import glob
import itertools
import json
import os
import sys


def walk_values(obj, key):
    """递归产出所有名为 key 的字段值（任意层级）。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                yield v
            else:
                yield from walk_values(v, key)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from walk_values(v, key)


def text_of(row):
    """尽量从各种结构里取出模型回答文本。"""
    for k in ("response", "prediction", "model_output", "output", "gen", "text"):
        v = row.get(k)
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            for kk in ("text", "content", "response"):
                if isinstance(v.get(kk), str):
                    return v[kk]
    return ""


def load(path):
    rows = []
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    else:
        try:
            data = json.load(open(path, encoding="utf-8", errors="replace"))
        except Exception:
            return rows
        if isinstance(data, list):
            rows = [r for r in data if isinstance(r, dict)]
        elif isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    rows.extend(v)
    return rows


def report(path):
    rows = load(path)
    if not rows:
        return False
    print("\n" + "=" * 70)
    print(path, "(%d rows)" % len(rows))
    print("keys:", ", ".join(sorted(rows[0].keys()))[:500])

    # finish_reason：递归搜集
    for key in ("finish_reason", "finish_reasons", "stop_reason"):
        vals = list(itertools.chain.from_iterable(walk_values(r, key) for r in rows))
        if vals:
            print("  %-14s %s" % (key + ":", dict(collections.Counter(str(v) for v in vals))))

    # 错误
    errs = [r.get("error") for r in rows if r.get("error")]
    print("  error:        %d 条" % len(errs))
    if errs:
        print("     示例:", str(errs[0])[:200])

    # 得分
    for key in ("score", "correct", "accuracy", "is_correct"):
        vals = [r.get(key) for r in rows if key in r]
        if vals:
            print("  %-14s %s" % (key + ":", dict(collections.Counter(str(v) for v in vals).most_common(6))))

    # 答案完整性（gsm8k 要求 "#### 数字"）
    texts = [text_of(r) for r in rows]
    n_box = sum(1 for t in texts if ("####" in t or "\\boxed" in t))
    print("  含 ####/\\boxed: %d/%d (%.1f%%)" % (n_box, len(rows), 100.0 * n_box / max(len(rows), 1)))

    lens = sorted(len(t) for t in texts if t)
    if lens:
        def q(f):
            return lens[min(int(len(lens) * f), len(lens) - 1)]
        print("  响应长度 chars: min=%d p50=%d p90=%d max=%d" % (lens[0], q(0.5), q(0.9), lens[-1]))

    # 错题预览（答案末尾，判断是"算错"还是"截断/没给最终答案"）
    wrong = []
    for r in rows:
        for key in ("score", "correct", "is_correct"):
            if key in r:
                v = str(r[key]).lower()
                if v in ("false", "0", "0.0", "incorrect"):
                    wrong.append(r)
                break
    if wrong:
        print("  错题数: %d（前 2 条答案尾部）" % len(wrong))
        for r in wrong[:2]:
            t = text_of(r).replace("\n", " ")
            print("    ...%s" % t[-260:])
    return True


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else None
    if root is None:
        cands = sorted(glob.glob("evalscope_out_*")) or sorted(glob.glob("outputs"))
        if not cands:
            print("没找到 evalscope_out_* / outputs/，请在执行 eval 的目录下运行，或把 work_dir 传进来")
            return 1
        root = cands[-1]
    print("work_dir:", root)

    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith((".jsonl", ".json")):
                p = os.path.join(dp, fn)
                files.append((os.path.getsize(p), p))
    files.sort(reverse=True)
    if not files:
        print("目录里没有 .jsonl/.json，实际内容：")
        for dp, dns, fns in os.walk(root):
            for fn in fns:
                print("   ", os.path.join(dp, fn))
        return 1

    hit = False
    for _, p in files:
        hit = report(p) or hit
    if not hit:
        print("\n没有可解析的评测记录，文件清单见上。")
        return 1
    print("\n" + "=" * 70)
    print("判据：finish_reason 出现 length ⇒ 被 max_tokens 截断；含 #### 比例低 ⇒ 答案格式/截断问题。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
