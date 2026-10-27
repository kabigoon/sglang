"""Branch-only trace for the DeepSeek-V4 learning sessions.

WHY THIS FILE EXISTS
--------------------
This is a *teaching* trace, not a production facility: it prints the shapes and
the small metadata tensors that let a human follow one DSV4 forward pass
(attention variant per layer -> mHC residual -> MoE dispatch/runner/combine).
It is env gated, writes one log per rank, and must never be merged to main.

ENV
---
    SGLANG_DSV4_TRACE=1                     master switch (default off)
    SGLANG_DSV4_TRACE_DIR=/tmp/dsv4_trace   log/dump root
    SGLANG_DSV4_TRACE_RANKS=0,1             only these global ranks (empty = all)
    SGLANG_DSV4_TRACE_LAYERS=0,4,20         only these layer ids (empty = all)
    SGLANG_DSV4_TRACE_LIMIT=2               hits per (tag, layer); 0 = unlimited
    SGLANG_DSV4_TRACE_DUMP=moe_dispatch_out tags whose tensors are dumped
    SGLANG_DSV4_TRACE_DUMP_LAYERS=0,4,20    layer filter for dumps (empty = all)

Output layout
-------------
    <DIR>/r{rank}.log
    <DIR>/dump/r{rank}/{step:05d}_{tag}_L{layer:02d}.pt

Design rules (learned the hard way in the ``none`` round):
  * never raise: a broken debug line must not kill the server, so every public
    helper swallows its own exceptions after warning once;
  * never sync a big tensor: shapes/dtypes come from metadata only, value
    summaries are restricted to small tensors (topk ids, expert counts);
  * call sites stay one-liners (``if dsv4_trace.ON: dsv4_trace.trace(...)``) so
    the injected diff stays reviewable and easy to revert.
"""

from __future__ import annotations

import os
import time

import torch

_ON = os.environ.get("SGLANG_DSV4_TRACE", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
ON = _ON

DIR = os.environ.get("SGLANG_DSV4_TRACE_DIR", "/tmp/dsv4_trace")


def _parse_ints(name: str):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    raw = raw.replace(" ", ",")
    return {int(p) for p in raw.split(",") if p}


_RANKS = _parse_ints("SGLANG_DSV4_TRACE_RANKS")
_LAYERS = _parse_ints("SGLANG_DSV4_TRACE_LAYERS")
_DUMP_TAGS = {
    t.strip()
    for t in os.environ.get("SGLANG_DSV4_TRACE_DUMP", "").split(",")
    if t.strip()
}
_DUMP_LAYERS = _parse_ints("SGLANG_DSV4_TRACE_DUMP_LAYERS")
try:
    LIMIT = int(os.environ.get("SGLANG_DSV4_TRACE_LIMIT", "0") or 0)
except ValueError:
    LIMIT = 0

_MAX_DUMP_NUMEL = 1 << 26  # 64M elements; anything bigger is skipped, not synced

_step = 0
_hits: dict[tuple[str, int], int] = {}
_rank: int | None = None
_warned = False


def _get_rank() -> int:
    global _rank
    if _rank is not None:
        return _rank
    rank = 0
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
    except Exception:
        pass
    _rank = int(rank)
    return _rank


def _fmt(v) -> str:
    """One-line description of any value: shape/dtype for tensors, repr otherwise."""
    if v is None:
        return "None"
    if isinstance(v, torch.Tensor):
        shape = "x".join(str(s) for s in v.shape)
        dtype = str(v.dtype).replace("torch.", "")
        return f"{v.dim()}d[{shape}]{dtype}"
    if isinstance(v, str):
        # Already-rendered strings (``describe`` output, enum names) stay bare:
        # quoting them would make the log harder to scan.
        return v if len(v) <= 200 else v[:197] + "..."
    if isinstance(v, (bool, int, float)):
        text = repr(v)
        return text if len(text) <= 48 else text[:45] + "..."
    if isinstance(v, (list, tuple)):
        if len(v) <= 6:
            return "[" + ",".join(_fmt(x) for x in v) + "]"
        return f"seq[{len(v)}]"
    if isinstance(v, dict):
        return "{" + ",".join(list(v)[:6]) + "}"
    name = type(v).__name__
    return f"<{name}>"


def _fields(obj) -> dict:
    """Best-effort field view of NamedTuple / dataclass / plain object."""
    if obj is None:
        return {}
    if hasattr(obj, "_asdict"):
        try:
            return dict(obj._asdict())
        except Exception:
            return {}
    if hasattr(obj, "__dataclass_fields__"):
        return {k: getattr(obj, k, None) for k in obj.__dataclass_fields__}
    data = getattr(obj, "__dict__", None)
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if not k.startswith("_")}
    return {}


def describe(obj, max_fields: int = 8) -> str:
    """Structural description of a dispatch/combine payload (backend agnostic)."""
    fields = _fields(obj)
    if not fields:
        return _fmt(obj)
    parts = []
    for i, (k, v) in enumerate(fields.items()):
        if i >= max_fields:
            parts.append("...")
            break
        parts.append(f"{k}={_fmt(v)}")
    return f"{type(obj).__name__}({', '.join(parts)})"


def values(v, head: int = 8) -> str:
    """Small value summary; only for tiny tensors (forces a device sync)."""
    if not isinstance(v, torch.Tensor) or v.numel() > 4096:
        return _fmt(v)
    try:
        flat = v.detach().reshape(-1).tolist()
    except Exception:
        return _fmt(v)
    shown = ",".join(str(x) for x in flat[:head])
    tail = ",..." if flat.__len__() > head else ""
    return f"[{shown}{tail}]"


def bincount(ids, num_experts: int | None = None, topk: int = 6) -> str:
    """Load summary of routed expert ids: sum / empty / busiest experts."""
    if not isinstance(ids, torch.Tensor):
        return _fmt(ids)
    try:
        flat = ids.detach().reshape(-1).long()
        if flat.numel() > 1 << 20:
            return _fmt(ids)
        n = int(num_experts) if num_experts else int(flat.max().item()) + 1
        counts = torch.bincount(flat, minlength=n)
        total = int(counts.sum().item())
        empty = int((counts == 0).sum().item())
        k = min(topk, counts.numel())
        top = torch.topk(counts, k)
        pairs = ",".join(
            f"{int(e)}:{int(c)}" for e, c in zip(top.indices.tolist(), top.values.tolist())
        )
        return f"rows={total}/E{n}/empty={empty}/top={pairs}"
    except Exception:
        return _fmt(ids)


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _admit(tag: str, layer: int) -> bool:
    if _RANKS is not None and _get_rank() not in _RANKS:
        return False
    if _LAYERS is not None and layer not in _LAYERS:
        return False
    key = (tag, layer)
    hits = _hits.get(key, 0) + 1
    _hits[key] = hits
    return not (LIMIT and hits > LIMIT)


def _write(line: str) -> None:
    _ensure_dir(DIR)
    path = os.path.join(DIR, f"r{_get_rank():02d}.log")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def trace(tag: str, layer: int = -1, **fields) -> None:
    """Append one line: ``[dsv4] r00 t00003 layer_in L03 k=v ...``."""
    global _step, _warned
    if not ON:
        return
    try:
        if not _admit(tag, layer):
            return
        _step += 1
        parts = [f"[dsv4] r{_get_rank():02d} t{_step:05d} {tag:16s} L{layer:02d}"]
        for k, v in fields.items():
            parts.append(f"{k}={_fmt(v)}")
        _write(" ".join(parts))
    except Exception as exc:  # never break the server for a debug line
        if not _warned:
            _warned = True
            print(f"[dsv4_trace] disabled after error: {exc!r}", flush=True)


def tensors_of(obj, max_fields: int = 16) -> dict:
    """Tensor fields of a dispatch/combine payload, flattened one level."""
    out: dict = {}
    fields = _fields(obj)
    if not fields:
        return out
    for i, (k, v) in enumerate(fields.items()):
        if i >= max_fields:
            break
        if isinstance(v, torch.Tensor):
            out[k] = v
        elif hasattr(v, "_fields"):
            out.update(tensors_of(v, max_fields))
    return out


def dump(
    tag: str, layer: int = -1, meta: dict | None = None, obj=None, **tensors
) -> None:
    """Save small tensors for one event (opt-in via SGLANG_DSV4_TRACE_DUMP)."""
    global _warned
    if not ON or tag not in _DUMP_TAGS:
        return
    try:
        if _DUMP_LAYERS is not None and layer not in _DUMP_LAYERS:
            return
        if obj is not None:
            tensors = {**tensors_of(obj), **tensors}
        kept, skipped = {}, []
        for k, v in tensors.items():
            if isinstance(v, torch.Tensor):
                if v.numel() > _MAX_DUMP_NUMEL:
                    skipped.append(f"{k}({tuple(v.shape)})")
                    continue
                kept[k] = v.detach().cpu()
            else:
                kept[k] = v
        if not kept:
            return
        out_dir = os.path.join(DIR, "dump", f"r{_get_rank():02d}")
        _ensure_dir(out_dir)
        name = f"{_step:05d}_{tag}_L{layer:02d}.pt"
        torch.save(
            {
                "tag": tag,
                "layer": layer,
                "step": _step,
                "rank": _get_rank(),
                "time": time.time(),
                "meta": meta or {},
                "skipped": skipped,
                **kept,
            },
            os.path.join(out_dir, name),
        )
    except Exception as exc:
        if not _warned:
            print(f"[dsv4_trace] dump disabled after error: {exc!r}", flush=True)
