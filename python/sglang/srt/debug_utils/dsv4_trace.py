"""Branch-only trace for the DeepSeek-V4 learning sessions (metadata only).

WHY THIS FILE EXISTS
--------------------
This is a *teaching* trace, not a production facility: it prints the shapes and
the small structural facts that let a human follow one DSV4 forward pass
(attention variant per layer -> mHC residual -> MoE dispatch/runner/combine).
It is env gated, writes one log per rank, and must never be merged to main.

HARD RULE: METADATA ONLY
------------------------
The first attempt read tensor *values* (``.tolist()`` / ``torch.bincount``) to
print expert load. On Ascend that ended in ``NPU out of memory. Tried to
allocate more than 1EB``: bincount sizes its output from ``max(id) + 1``, i.e. a
size computed from tensor contents, which is not a safe thing to do inside a
forward. This module therefore only ever touches ``.shape`` / ``.dtype`` /
``.dim()`` and python attributes. If you add a value read, you own the next OOM.

ENV
---
    SGLANG_DSV4_TRACE=1                     master switch (default off)
    SGLANG_DSV4_TRACE_DIR=/tmp/dsv4_trace   log root
    SGLANG_DSV4_TRACE_RANKS=0,1             only these global ranks (empty = all)
    SGLANG_DSV4_TRACE_LAYERS=0,4,20         only these layer ids (empty = all)
    SGLANG_DSV4_TRACE_LIMIT=1               hits per (tag, layer); 0 = unlimited

Output: ``<DIR>/r{rank:02d}.log``, one line per event:

    [dsv4] r00 t00002 layer_in         L00 R=3d[7x4x4096]bf16 cr=0 cmp=False ...

Design rules:
  * never raise: a broken debug line must not kill the server, so the writer
    swallows its own exceptions after warning once;
  * call sites stay one-liners (``if dsv4_trace.ON: dsv4_trace.trace(...)``) so
    the injected diff stays reviewable and easy to revert.
"""

from __future__ import annotations

import os

import torch

ON = os.environ.get("SGLANG_DSV4_TRACE", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
DIR = os.environ.get("SGLANG_DSV4_TRACE_DIR", "/tmp/dsv4_trace")
# Opt-in value peeks. Only ever applied to TINY tensors (<= _MAX_VALUE_ELEMS)
# and always copied to CPU first: no size is derived from tensor contents.
ON_VALUES = os.environ.get("SGLANG_DSV4_TRACE_VALUES", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_MAX_VALUE_ELEMS = 256


def _parse_ints(name: str):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    raw = raw.replace(" ", ",")
    return {int(p) for p in raw.split(",") if p}


_RANKS = _parse_ints("SGLANG_DSV4_TRACE_RANKS")
_LAYERS = _parse_ints("SGLANG_DSV4_TRACE_LAYERS")
try:
    LIMIT = int(os.environ.get("SGLANG_DSV4_TRACE_LIMIT", "0") or 0)
except ValueError:
    LIMIT = 0

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
    """One-line description of any value. Reads metadata only, never data."""
    if v is None:
        return "None"
    if isinstance(v, torch.Tensor):
        shape = "x".join(str(s) for s in v.shape)
        dtype = str(v.dtype).replace("torch.", "")
        return f"{v.dim()}d[{shape}]{dtype}"
    if isinstance(v, str):
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
    return f"<{type(v).__name__}>"


def vals(t, n: int = 8) -> str:
    """Peek at a TINY tensor's values (opt-in via SGLANG_DSV4_TRACE_VALUES).

    Deliberately limited to ``_MAX_VALUE_ELEMS`` elements and always copied to
    CPU before touching the data, so no allocation size depends on contents.
    Returns "" when disabled or unsupported, which ``trace`` drops silently.
    """
    if not ON_VALUES or not isinstance(t, torch.Tensor):
        return ""
    try:
        if t.numel() == 0:
            return "empty"
        if t.numel() > _MAX_VALUE_ELEMS:
            return f"big[{t.numel()}]"
        flat = t.detach().to("cpu").reshape(-1)
        head = flat[:n].tolist()
        body = ",".join(str(x) for x in head)
        more = "" if flat.numel() <= n else f",...({flat.numel()})"
        return f"[{body}{more}]"
    except Exception:
        return ""


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


def _admit(tag: str, layer: int) -> bool:
    if _RANKS is not None and _get_rank() not in _RANKS:
        return False
    # layer < 0 marks a model-level event (model_in / model_out): it is never
    # filtered by --layers, otherwise a layer-scoped run loses the batch shape.
    if _LAYERS is not None and layer >= 0 and layer not in _LAYERS:
        return False
    key = (tag, layer)
    hits = _hits.get(key, 0) + 1
    _hits[key] = hits
    return not (LIMIT and hits > LIMIT)


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
            if v == "":
                continue  # value peeks that are disabled / unsupported
            parts.append(f"{k}={_fmt(v)}")
        os.makedirs(DIR, exist_ok=True)
        with open(os.path.join(DIR, f"r{_get_rank():02d}.log"), "a") as fh:
            fh.write(" ".join(parts) + "\n")
    except Exception as exc:  # never break the server for a debug line
        if not _warned:
            _warned = True
            print(f"[dsv4_trace] disabled after error: {exc!r}", flush=True)
