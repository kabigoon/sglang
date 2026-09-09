#!/usr/bin/env python3
"""Standalone A5 DSpark sparse-attention smoke test with ELF provenance output.

The script re-executes itself before importing torch so LD_LIBRARY_PATH and
ASCEND_CUSTOM_OPP_PATH are visible while the dynamic loader initializes.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import traceback


OLD_TILING_MARKER = b"oriSparseIndices is not supported now"
NEW_TILING_MARKER = b"cuSeqLensOriKv is not supported now"
LIBRARY_MARKERS = (
    "vllm_ascend_c",
    "libcust_opapi",
    "libcust_opmaster",
    "libcust_opsproto",
    "liboptiling",
    "custom_transformer",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vllm-root",
        default="/home/a00821909/vllm-ascend",
        help="vllm-ascend checkout containing build/ and vllm_ascend/_cann_ops_custom/",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--draft-tokens", type=int, default=5)
    parser.add_argument(
        "--ld-debug",
        action="store_true",
        help="Write the dynamic loader trace to /tmp/dspark-a5-lddebug.<pid>",
    )
    return parser.parse_args()


def prepend_unique(value: str, current: str | None) -> str:
    paths = [value]
    if current:
        paths.extend(path for path in current.split(":") if path and path != value)
    return ":".join(paths)


def configure_and_reexec(args: argparse.Namespace) -> None:
    if os.environ.get("_DSPARK_A5_SMOKE_REEXEC") == "1":
        return

    root = Path(args.vllm_root).resolve()
    vendor = root / "vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
    binding_candidates = sorted((root / "build").glob("vllm_ascend_C*.so"))
    if len(binding_candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one {root}/build/vllm_ascend_C*.so, "
            f"found {[str(path) for path in binding_candidates]}"
        )

    env = os.environ.copy()
    env["_DSPARK_A5_SMOKE_REEXEC"] = "1"
    env["ASCEND_CUSTOM_OPP_PATH"] = str(vendor)
    env["LD_LIBRARY_PATH"] = prepend_unique(
        str(vendor / "op_api/lib"),
        prepend_unique(str(root / "build"), env.get("LD_LIBRARY_PATH")),
    )
    env["SGLANG_DSPARK_A5_EXTRA_OPS_SO"] = str(binding_candidates[0])
    if args.ld_debug:
        env["LD_DEBUG"] = "libs,files"
        env["LD_DEBUG_OUTPUT"] = "/tmp/dspark-a5-lddebug"
    os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)


def contains_marker(path: Path, marker: bytes) -> bool:
    overlap = max(len(marker) - 1, 0)
    tail = b""
    try:
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                data = tail + chunk
                if marker in data:
                    return True
                tail = data[-overlap:] if overlap else b""
    except OSError:
        return False
    return False


def mapped_libraries() -> list[Path]:
    paths: set[Path] = set()
    for line in Path("/proc/self/maps").read_text(errors="replace").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6 or not fields[5].startswith("/"):
            continue
        raw_path = fields[5].removesuffix(" (deleted)")
        if any(marker in raw_path.lower() for marker in LIBRARY_MARKERS):
            paths.add(Path(raw_path))
    return sorted(paths, key=str)


def print_library_provenance(stage: str, torch) -> None:
    print(f"\n===== shared-library provenance: {stage} =====", flush=True)
    print("binding env:", os.getenv("SGLANG_DSPARK_A5_EXTRA_OPS_SO"), flush=True)
    print("ASCEND_CUSTOM_OPP_PATH:", os.getenv("ASCEND_CUSTOM_OPP_PATH"), flush=True)
    print("LD_LIBRARY_PATH:", os.getenv("LD_LIBRARY_PATH"), flush=True)
    loaded = getattr(torch.ops, "loaded_libraries", None)
    if isinstance(loaded, set):
        print("torch.ops.loaded_libraries:", sorted(loaded), flush=True)

    paths = mapped_libraries()
    if not paths:
        print("/proc/self/maps: no matching libraries", flush=True)
    for path in paths:
        print(
            f"mapped={path} "
            f"owner={path.stat().st_uid if path.exists() else 'missing'} "
            f"old_ori_rejection={contains_marker(path, OLD_TILING_MARKER)} "
            f"new_cuseq_rejection={contains_marker(path, NEW_TILING_MARKER)}",
            flush=True,
        )


def show_schema(torch, name: str):
    op = getattr(torch.ops._C_ascend, name)
    print(f"\n{name}:\n{op.default._schema}", flush=True)
    return op


def run_test(args: argparse.Namespace) -> None:
    import torch
    import torch_npu  # noqa: F401  registers the NPU dispatch key

    assert torch.npu.is_available(), "PyTorch did not detect an NPU"
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")

    binding = Path(os.environ["SGLANG_DSPARK_A5_EXTRA_OPS_SO"])
    torch.ops.load_library(str(binding))
    print("PID:", os.getpid())
    if ld_debug_output := os.getenv("LD_DEBUG_OUTPUT"):
        print("dynamic-loader trace:", f"{ld_debug_output}.{os.getpid()}")
    print("PyTorch:", torch.__version__)
    print("NPU:", torch.npu.get_device_name(args.device))
    print_library_provenance("after torch.ops.load_library", torch)

    metadata_op = show_schema(torch, "npu_kv_quant_sparse_attn_sharedkv_metadata")
    attention_op = show_schema(torch, "npu_kv_quant_sparse_attn_sharedkv")

    batch_size = 1
    num_q_heads = 64
    num_kv_heads = 1
    head_dim = 512
    packed_kv_dim = 640
    page_size = 128
    query_len = args.draft_tokens
    kv_len = args.window_size + args.draft_tokens
    ori_win_left = args.window_size + args.draft_tokens - 1
    index_width = math.ceil(kv_len / 128) * 128
    num_pages = math.ceil(kv_len / page_size)

    cu_seqlens_q = torch.tensor([0, query_len], dtype=torch.int32, device=device)
    seqused_kv = torch.tensor([kv_len], dtype=torch.int32, device=device)

    metadata = metadata_op(
        num_heads_q=num_q_heads,
        num_heads_kv=num_kv_heads,
        head_dim=head_dim,
        kv_quant_mode=1,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        batch_size=batch_size,
        max_seqlen_q=query_len,
        max_seqlen_kv=kv_len,
        ori_topk=0,
        cmp_topk=0,
        tile_size=64,
        rope_head_dim=64,
        cmp_ratio=1,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=ori_win_left,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
        has_ori_kv=True,
        has_cmp_kv=False,
        device=str(device),
    )
    torch.npu.synchronize()
    print("\n[PASS] metadata op", tuple(metadata.shape), metadata.dtype, metadata.device)
    print_library_provenance("after metadata op", torch)

    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        raise RuntimeError("torch.float8_e4m3fn is unavailable")

    q = torch.zeros((query_len, num_q_heads, head_dim), dtype=torch.bfloat16, device=device)
    ori_kv = torch.zeros(
        (num_pages, page_size, num_kv_heads, packed_kv_dim),
        dtype=fp8_dtype,
        device=device,
    )
    ori_block_table = torch.arange(num_pages, dtype=torch.int32, device=device).view(1, -1)
    visible_slots = torch.arange(index_width, dtype=torch.int32, device=device)
    visible_slots = torch.where(
        visible_slots < kv_len,
        visible_slots,
        torch.full_like(visible_slots, -1),
    )
    ori_sparse_indices = visible_slots.view(1, 1, index_width).expand(
        query_len, num_kv_heads, index_width
    ).contiguous()
    sinks = torch.zeros(num_q_heads, dtype=torch.float32, device=device)

    try:
        out, softmax_lse = attention_op(
            q,
            kv_quant_mode=1,
            ori_kv=ori_kv,
            ori_sparse_indices=ori_sparse_indices,
            ori_block_table=ori_block_table,
            cu_seqlens_q=cu_seqlens_q,
            seqused_kv=seqused_kv,
            sinks=sinks,
            metadata=metadata,
            tile_size=64,
            rope_head_dim=64,
            softmax_scale=1.0 / math.sqrt(head_dim),
            cmp_ratio=1,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=ori_win_left,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            return_softmax_lse=False,
        )
        torch.npu.synchronize()
    except BaseException:
        print_library_provenance("after attention failure", torch)
        raise

    print_library_provenance("after attention success", torch)
    print("\n[PASS] DSpark attention op")
    print("out:", tuple(out.shape), out.dtype, out.device)
    print("softmax_lse:", tuple(softmax_lse.shape), softmax_lse.dtype)
    print("finite:", torch.isfinite(out.float()).all().item())
    assert out.shape == q.shape
    assert out.dtype == torch.bfloat16


def main() -> int:
    args = parse_args()
    try:
        configure_and_reexec(args)
        run_test(args)
    except BaseException:
        traceback.print_exc()
        return 1
    print("\nAll A5 DSpark custom-op smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
