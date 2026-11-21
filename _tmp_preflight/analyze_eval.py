#!/usr/bin/env python3
"""分析 evalscope 评测产物（已按 evalscope 1.15 实测 schema 适配）。

实测结构（2026-11-21, gsm8k / openai_api）：
  predictions/*.jsonl  行 = {agent_trace, index, messages, metadata, model, model_output}
    model_output = {id, model, choices: [{logprobs, message, stop_reason}]}
    message      = {content, id, internal, metadata, model, perf_metrics, role, source, tool_calls}
    ⚠ 是 choices[0].stop_reason，不是 finish_reason；思考内容合并进 content（`</think>` 分隔）
  reviews/*.jsonl      行 = {agent_trace, index, messages, sample_score, target}
    sample_score.score = {value: {accuracy}, status, extracted_prediction, prediction, ...}
    target = ["109"]    ← 列表，不是字符串
  reports/*.json        行 = {categories, identity, legacy_name, macro_score, num, num_correct, score}

用法（在当初执行 eval_gsm8k.sh 的目录下）：
    python3 _tmp_preflight/analyze_eval.py [work_dir]
"""
import collections
import glob
import json
import os
import sys


# ---------------------------------------------------------------- 取值
def norm_gold(t):
    if isinstance(t, (list, tuple)):
        return "/".join(str(x) for x in t)
    return "" if t is None else str(t)


def pred_fields(row):
    """(答案文本, stop_reason)"""
    mo = row.get("model_output")
    text, stop = "", None
    if isinstance(mo, dict):
        choices = mo.get("choices") or []
        if choices and isinstance(choices[0], dict):
            ch = choices[0]
            stop = ch.get("stop_reason")
            msg = ch.get("message") or {}
            if isinstance(msg, dict):
                text = msg.get("content") or ""
    if not text:  # 兜底：任意嵌套里的字符串
        text = _deep_text(mo)
    if stop is None:
        stop = (row.get("metadata") or {}).get("stop_reason")
    return text, stop


def review_fields(row):
    """(accuracy, extracted_prediction)"""
    ss = row.get("sample_score") or {}
    sc = (ss.get("score") if isinstance(ss, dict) else None) or {}
    val = sc.get("value") or {}
    return val.get("accuracy"), sc.get("extracted_prediction")


def _deep_text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return "\n".join(t for t in (_deep_text(v) for v in x.values()) if t)
    if isinstance(x, (list, tuple)):
        return "\n".join(t for t in (_deep_text(v) for v in x) if t)
    return ""


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def load_json(path):
    try:
        data = json.load(open(path, encoding="utf-8", errors="replace"))
    except Exception:
        return []
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        return [data]
    return []


# ---------------------------------------------------------------- 报告
def q(vals, f):
    vals = sorted(vals)
    return vals[min(int(len(vals) * f), len(vals) - 1)] if vals else 0


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else None
    if root is None:
        cands = sorted(glob.glob("evalscope_out_*")) or sorted(glob.glob("outputs"))
        if not cands:
            print("没找到 evalscope_out_*；请在执行 eval 的目录下运行，或传入 work_dir")
            return 1
        root = cands[-1]
    print("work_dir:", root)

    pred_files = sorted(glob.glob(os.path.join(root, "**", "predictions", "**", "*.jsonl"), recursive=True))
    rev_files = sorted(glob.glob(os.path.join(root, "**", "reviews", "**", "*.jsonl"), recursive=True))
    rep_files = sorted(glob.glob(os.path.join(root, "**", "reports", "**", "*.json"), recursive=True))
    if not pred_files and not rev_files:
        for dp, _, fns in os.walk(root):
            for fn in fns:
                print("   ", os.path.join(dp, fn))
        print("没找到 predictions/reviews，先把上面的文件清单发出来")
        return 1

    preds = {}
    for f in pred_files:
        for i, r in enumerate(load_jsonl(f)):
            key = r.get("index", i)
            preds[key] = r
    revs = {}
    for f in rev_files:
        for i, r in enumerate(load_jsonl(f)):
            key = r.get("index", i)
            revs[key] = r

    print("predictions: %d 条 %s" % (len(preds), [os.path.basename(x) for x in pred_files]))
    print("reviews    : %d 条 %s" % (len(revs), [os.path.basename(x) for x in rev_files]))

    # ---- 逐条合并
    records = []
    for key in sorted(set(preds) | set(revs)):
        p, r = preds.get(key, {}), revs.get(key, {})
        text, stop = pred_fields(p) if p else ("", None)
        acc, ext = review_fields(r) if r else (None, None)
        gold = norm_gold(r.get("target")) if r else ""
        records.append({"index": key, "text": text, "stop": stop, "acc": acc,
                        "ext": ext, "gold": gold})

    # ---- 汇总
    print("\n" + "=" * 72)
    stops = collections.Counter(str(x["stop"]) for x in records if x["stop"] is not None)
    if stops:
        print("stop_reason :", dict(stops))
    n_known = sum(1 for x in records if x["acc"] is not None)
    n_ok = sum(1 for x in records if x["acc"] not in (None, 0, 0.0))
    if n_known:
        print("准确率      : %d/%d = %.2f%%" % (n_ok, n_known, 100.0 * n_ok / n_known))
    boxed = sum(1 for x in records if ("####" in x["text"] or "\\boxed" in x["text"]))
    print("含 boxed/####: %d/%d" % (boxed, len(records)))
    lens = [len(x["text"]) for x in records if x["text"]]
    if lens:
        print("答案长度    : min=%d p50=%d p90=%d max=%d" % (lens and min(lens), q(lens, 0.5), q(lens, 0.9), max(lens)))
    trunc = [x for x in records if str(x["stop"]) not in ("stop", "None")]
    print("非正常结束  : %d 条 %s" % (len(trunc), collections.Counter(str(x["stop"]) for x in trunc) or ""))

    # ---- reports 交叉验证
    for f in rep_files:
        for r in load_json(f):
            if "num" in r and "num_correct" in r:
                print("reports     : %s -> correct/num=%s/%s score=%s" % (
                    os.path.basename(f), r.get("num_correct"), r.get("num"), r.get("score")))

    # ---- 错题表
    wrong = [x for x in records if x["acc"] in (0, 0.0)]
    if wrong:
        print("\n" + "-" * 72)
        print("错题 %d 条：" % len(wrong))
        print("  %-5s %-14s %-14s %-6s %s" % ("idx", "抽取值", "gold", "stop", "答案尾部"))
        for x in wrong:
            print("  %-5s %-14s %-14s %-6s %s" % (
                x["index"],
                str(x["ext"])[:14],
                x["gold"][:14],
                str(x["stop"]),
                x["text"][-90:].replace("\n", " "),
            ))
        mismatch = sum(1 for x in wrong if x["ext"] and x["gold"] and str(x["ext"]) == x["gold"])
        if mismatch:
            print("\n  ⚠ 其中 %d 条抽取值与 gold 相同却判错 → 判分/抽取问题，不是算错" % mismatch)
    else:
        print("\n没有错题（或 reviews 里没有 accuracy 字段）。")

    print("\n判据：stop 全是 stop ⇒ 没截断；错题里『抽取值==gold 却判错』⇒ 判分问题，需查 target 解析。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
