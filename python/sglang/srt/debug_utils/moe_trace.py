"""本机调试用 MoE 打点工具（临时文件，不要提交）。

目的：在不动业务逻辑的前提下，观察 MoE 链路每步的张量形状 / dtype / 设备，
为 ascend_tp → None 迁移建立"改动前"的参照。

用法::

    export SGLANG_MOE_TRACE=1
    export SGLANG_MOE_TRACE_LAYERS=1,13     # 只打这几层（不设 = 所有层）
    export SGLANG_MOE_TRACE_LIMIT=3         # 每个 (tag, layer, rank) 只记前 N 次；0 = 不限
    export SGLANG_MOE_TRACE_DIR=/tmp/moe_trace
    export SGLANG_MOE_DUMP_LAYERS=1         # 只对第 1 层做张量落盘（不设 = 不落盘）

默认全关：不设 ``SGLANG_MOE_TRACE`` 时本模块只有一次 import 开销，
``trace()`` / ``dump()`` 立即返回，不产生任何输出、也不改变任何行为。

日志：``<DIR>/rank_tp{rank}.log``（每个 TP rank 一个文件，避免多进程日志交错）
落盘：``<DIR>/dump/tp{rank}/L{layer_id}/{tag}.pt``
"""

from __future__ import annotations

import os
from typing import Any

import torch

_TRUE = ("1", "true", "True", "yes", "on")


def _on(name: str) -> bool:
    return os.environ.get(name, "0") in _TRUE


TRACE_ENABLED = _on("SGLANG_MOE_TRACE")
_LIMIT = int(os.environ.get("SGLANG_MOE_TRACE_LIMIT", "3"))
_DIR = os.environ.get("SGLANG_MOE_TRACE_DIR", "/tmp/moe_trace")
_TRACE_LAYERS = {
    int(x)
    for x in os.environ.get("SGLANG_MOE_TRACE_LAYERS", "").split(",")
    if x.strip()
}
_DUMP_LAYERS = {
    int(x) for x in os.environ.get("SGLANG_MOE_DUMP_LAYERS", "").split(",") if x.strip()
}

_counters: dict[tuple[str, int, Any], int] = {}


def _ranks() -> tuple[int, int, int, int]:
    try:
        from sglang.srt.runtime_context import get_parallel

        p = get_parallel()
        return (
            getattr(p, "tp_rank", -1),
            getattr(p, "attn_dp_rank", -1),
            getattr(p, "moe_ep_rank", -1),
            getattr(p, "moe_tp_rank", -1),
        )
    except Exception:  # 在 init/publish 之前被调用时不该炸
        return (-1, -1, -1, -1)


def fmt(v: Any) -> str:
    """张量只打 形状/dtype/设备；标量与容器原样展开。"""
    if isinstance(v, torch.Tensor):
        shape = tuple(v.shape) if v.dim() else (0,)
        dev = v.device.type if v.device is not None else "?"
        return f"{shape}:{str(v.dtype).replace('torch.', '')}:{dev}"
    if v is None:
        return "None"
    if isinstance(v, (tuple, list)):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}:{fmt(x)}" for k, x in v.items()) + "}"
    if isinstance(v, torch.dtype):
        return str(v).replace("torch.", "")
    if isinstance(v, (int, float, bool, str, bytes)):
        return repr(v)
    # Anything else (deferred "owed" handles, ForwardBatch, partials, ...) keeps
    # one token only: the class name is the information, the repr is megabytes.
    return f"<{type(v).__name__}>"


def trace(tag: str, **kw: Any) -> None:
    """记录一行打点。

    ``kw`` 里若带 ``layer=<int>``，则：受 ``SGLANG_MOE_TRACE_LAYERS`` 过滤，
    并按 (tag, layer, rank) 单独计数（便于"每层只看 N 次"）。
    """
    if not TRACE_ENABLED:
        return
    layer = kw.get("layer")
    if _TRACE_LAYERS and isinstance(layer, int) and layer not in _TRACE_LAYERS:
        return
    tp, dp, ep, mtp = _ranks()
    key = (tag, tp, layer)
    n = _counters.get(key, 0)
    _counters[key] = n + 1
    if _LIMIT and n >= _LIMIT:
        return
    body = " ".join(f"{k}={fmt(v)}" for k, v in kw.items())
    prefix = f"[tp{tp} dp{dp} ep{ep} mtp{mtp}][{tag}#{n}]"
    line = f"{prefix} {body}\n"
    os.makedirs(_DIR, exist_ok=True)
    with open(os.path.join(_DIR, f"rank_tp{tp}.log"), "a") as f:
        f.write(line)


def dump(layer_id: int, tag: str, **tensors: Any) -> None:
    """把指定层的张量落盘，用于迁移前后逐张量对比（golden dump）。

    只对 ``SGLANG_MOE_DUMP_LAYERS`` 里列出的层生效；未设置则全关。
    """
    if not TRACE_ENABLED or layer_id not in _DUMP_LAYERS:
        return
    tp, _, _, _ = _ranks()
    out_dir = os.path.join(_DIR, "dump", f"tp{tp}", f"L{layer_id}")
    os.makedirs(out_dir, exist_ok=True)
    payload = {}
    for k, v in tensors.items():
        if isinstance(v, torch.Tensor):
            payload[k] = v.detach().to("cpu")
        elif isinstance(v, (tuple, list)) and all(
            isinstance(x, torch.Tensor) for x in v
        ):
            payload[k] = [x.detach().to("cpu") for x in v]
    torch.save(payload, os.path.join(out_dir, f"{tag}.pt"))
