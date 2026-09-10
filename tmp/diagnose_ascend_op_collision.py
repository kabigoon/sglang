#!/usr/bin/env python3
"""Diagnose custom.compressor vs _C_ascend ACLNN library collisions.

Run this script on the NPU host.  The parent process deliberately does not
import torch; it starts every scenario in a fresh child so that dynamic-loader
"first load wins" effects can be compared reliably.

Example:
    python3 tmp/diagnose_ascend_op_collision.py \
        --vllm-root /home/a00821909/vllm-ascend \
        --scenario all

The inherited ASCEND_CUSTOM_OPP_PATH is retained after the vllm-ascend sparse
operator vendor.  Source the normal CANN/custom_ops environment before running
the script so the system compressor remains available.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from typing import Any


SPARSE_OPS = (
    "npu_kv_quant_sparse_attn_sharedkv_metadata",
    "npu_kv_quant_sparse_attn_sharedkv",
)
ACLNN_SYMBOLS = (
    "aclnnCompressorGetWorkspaceSize",
    "aclnnCompressor",
    "aclnnKvQuantSparseAttnSharedkvMetadataGetWorkspaceSize",
    "aclnnKvQuantSparseAttnSharedkvMetadata",
    "aclnnKvQuantSparseAttnSharedkvGetWorkspaceSize",
    "aclnnKvQuantSparseAttnSharedkv",
)
OLD_TILING_MARKER = b"oriSparseIndices is not supported now"
NEW_TILING_MARKER = b"cuSeqLensOriKv is not supported now"
MAP_MARKERS = (
    "vllm_ascend",
    "custom_ops",
    "cust_op",
    "opmaster",
    "opsproto",
    "optiling",
    "custom_transformer",
    "/vendors/customize/",
)
SCENARIOS = (
    "sparse-only",
    "custom-import-then-sparse",
    "compressor-then-sparse",
    "sparse-then-compressor-then-sparse",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose which CANN custom-op ELF objects back torch ops."
    )
    parser.add_argument(
        "--vllm-root",
        type=Path,
        default=Path("/home/a00821909/vllm-ascend"),
        help="vllm-ascend checkout containing build/ and _cann_ops_custom/",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        help="Explicit vllm_ascend_C*.so (otherwise discovered under build/)",
    )
    parser.add_argument(
        "--sparse-vendor",
        type=Path,
        help="Explicit sparse custom_transformer vendor directory",
    )
    parser.add_argument(
        "--scenario",
        choices=("all", *SCENARIOS),
        default="all",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--draft-tokens", type=int, default=5)
    parser.add_argument(
        "--custom-module",
        default="custom_ops",
        help="Module imported by SGLang init_npu_backend to register custom.* ops",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/ascend-op-collision"),
        help="Directory for per-scenario JSON reports and optional LD_DEBUG logs",
    )
    parser.add_argument(
        "--opp-mode",
        choices=("prepend", "only", "preserve"),
        default="prepend",
        help=(
            "prepend sparse vendor to inherited ASCEND_CUSTOM_OPP_PATH (default), "
            "use only sparse vendor, or preserve the inherited value"
        ),
    )
    parser.add_argument(
        "--ld-debug",
        action="store_true",
        help="Enable ld.so library/binding traces in the output directory",
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def prepend_unique(value: str, current: str | None) -> str:
    paths = [value]
    if current:
        paths.extend(item for item in current.split(":") if item and item != value)
    return ":".join(paths)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    root = args.vllm_root.expanduser().resolve()
    vendor = (
        args.sparse_vendor.expanduser().resolve()
        if args.sparse_vendor
        else root / "vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
    )
    if not vendor.is_dir():
        raise RuntimeError(f"Sparse operator vendor directory is missing: {vendor}")

    if args.binding:
        binding = args.binding.expanduser().resolve()
        candidates = [binding] if binding.is_file() else []
    else:
        candidates = sorted((root / "build").glob("vllm_ascend_C*.so"))
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one vllm_ascend_C*.so; found "
            f"{[str(item) for item in candidates]}"
        )
    return candidates[0], vendor


def child_environment(
    args: argparse.Namespace, binding: Path, vendor: Path, scenario: str
) -> dict[str, str]:
    env = os.environ.copy()
    inherited_opp = env.get("ASCEND_CUSTOM_OPP_PATH")
    if args.opp_mode == "prepend":
        env["ASCEND_CUSTOM_OPP_PATH"] = prepend_unique(str(vendor), inherited_opp)
    elif args.opp_mode == "only":
        env["ASCEND_CUSTOM_OPP_PATH"] = str(vendor)
    elif inherited_opp is None:
        env.pop("ASCEND_CUSTOM_OPP_PATH", None)

    env["LD_LIBRARY_PATH"] = prepend_unique(
        str(vendor / "op_api/lib"),
        prepend_unique(str(binding.parent), env.get("LD_LIBRARY_PATH")),
    )
    env["SGLANG_DSPARK_A5_EXTRA_OPS_SO"] = str(binding)
    env["_ASCEND_OP_COLLISION_SCENARIO"] = scenario
    if args.ld_debug:
        env["LD_DEBUG"] = "libs,bindings"
        env["LD_DEBUG_OUTPUT"] = str(args.output_dir / f"lddebug-{scenario}")
    return env


def child_command(args: argparse.Namespace, scenario: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--scenario",
        scenario,
        "--vllm-root",
        str(args.vllm_root),
        "--device",
        str(args.device),
        "--window-size",
        str(args.window_size),
        "--draft-tokens",
        str(args.draft_tokens),
        "--custom-module",
        args.custom_module,
        "--output-dir",
        str(args.output_dir),
        "--opp-mode",
        args.opp_mode,
    ]
    if args.binding:
        command.extend(("--binding", str(args.binding)))
    if args.sparse_vendor:
        command.extend(("--sparse-vendor", str(args.sparse_vendor)))
    return command


def elf_dynamic_tags(path: Path) -> list[str]:
    readelf = shutil.which("readelf")
    if readelf is None:
        return ["<readelf unavailable>"]
    try:
        completed = subprocess.run(
            [readelf, "-d", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return [f"<readelf failed: {exc!r}>"]
    tags = (
        line.strip()
        for line in completed.stdout.splitlines()
        if any(tag in line for tag in ("SONAME", "NEEDED", "RPATH", "RUNPATH"))
    )
    return list(tags)


def static_elf_inventory(vendor: Path) -> dict[str, Any]:
    roots = [vendor]
    inherited = os.getenv("ASCEND_CUSTOM_OPP_PATH")
    if inherited:
        roots.extend(Path(item) for item in inherited.split(":") if item)

    patterns = (
        "libcust_opapi.so*",
        "libcust_opmaster*.so*",
        "libcust_opsproto*.so*",
        "liboptiling*.so*",
    )
    candidates: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in patterns:
            try:
                candidates.update(path for path in root.rglob(pattern) if path.is_file())
            except OSError:
                continue

    files = []
    for path in sorted(candidates, key=str):
        item = describe_path(path)
        item["dynamic_tags"] = elf_dynamic_tags(path)
        files.append(item)
    return {"roots": [str(root) for root in roots], "files": files}


def run_parent(args: argparse.Namespace) -> int:
    binding, vendor = resolve_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = SCENARIOS if args.scenario == "all" else (args.scenario,)

    inventory = static_elf_inventory(vendor)
    inventory["binding"] = describe_path(binding) | {
        "dynamic_tags": elf_dynamic_tags(binding)
    }
    inventory_path = args.output_dir / "static-elf-inventory.json"
    inventory_path.write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("binding:", binding)
    print("sparse vendor:", vendor)
    print("inherited ASCEND_CUSTOM_OPP_PATH:", os.getenv("ASCEND_CUSTOM_OPP_PATH"))
    print("reports:", args.output_dir)
    print("static ELF inventory:", inventory_path)
    for item in inventory["files"]:
        sonames = [line for line in item["dynamic_tags"] if "SONAME" in line]
        print(
            f"ELF: {item['path']} SONAME={sonames or '<none>'} "
            f"old={item.get('old_tiling_marker')} "
            f"new={item.get('new_tiling_marker')}"
        )
    results: dict[str, int] = {}
    for scenario in scenarios:
        print(f"\n{'#' * 24} {scenario} {'#' * 24}", flush=True)
        env = child_environment(args, binding, vendor, scenario)
        completed = subprocess.run(child_command(args, scenario), env=env, check=False)
        results[scenario] = completed.returncode

    print("\n===== scenario exit codes =====")
    for scenario, returncode in results.items():
        print(f"{scenario}: {returncode}")
    print(
        "A non-zero collision scenario is still useful: inspect its JSON and "
        "the last library snapshot to see which implementation won."
    )
    # Do not stop a matrix after an expected collision.  A single requested
    # scenario preserves its exit status for automation.
    return next(iter(results.values())) if len(results) == 1 else 0


@lru_cache(maxsize=None)
def file_contains(path: Path, marker: bytes) -> bool:
    overlap = max(0, len(marker) - 1)
    tail = b""
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                data = tail + chunk
                if marker in data:
                    return True
                tail = data[-overlap:] if overlap else b""
    except OSError:
        return False
    return False


@lru_cache(maxsize=None)
def short_hash(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            # Identity is enough here; avoid hashing very large device binaries.
            digest.update(stream.read(4 * 1024 * 1024))
        return digest.hexdigest()[:16]
    except OSError:
        return None


def mapped_libraries() -> list[Path]:
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return []
    result: set[Path] = set()
    for line in maps.read_text(errors="replace").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6 or not fields[5].startswith("/"):
            continue
        raw = fields[5].removesuffix(" (deleted)")
        if any(marker in raw.lower() for marker in MAP_MARKERS):
            result.add(Path(raw))
    return sorted(result, key=str)


class DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


def global_symbol_owners() -> dict[str, str | None]:
    """Resolve process-global symbols; this often exposes the first-loaded API."""
    owners: dict[str, str | None] = {}
    libdl_name = ctypes.util.find_library("dl")
    libdl = ctypes.CDLL(libdl_name) if libdl_name else ctypes.CDLL(None)
    dladdr = libdl.dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    dladdr.restype = ctypes.c_int
    process = ctypes.CDLL(None)
    for symbol in ACLNN_SYMBOLS:
        try:
            function = getattr(process, symbol)
            address = ctypes.cast(function, ctypes.c_void_p)
            info = DlInfo()
            if dladdr(address, ctypes.byref(info)) and info.dli_fname:
                owners[symbol] = os.fsdecode(info.dli_fname)
            else:
                owners[symbol] = "<resolved, dladdr unavailable>"
        except (AttributeError, OSError):
            owners[symbol] = None
    return owners


def describe_path(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
        return {
            "path": str(path),
            "realpath": str(path.resolve()),
            "size": stat.st_size,
            "inode": stat.st_ino,
            "first_4m_sha256": short_hash(path),
            "old_tiling_marker": file_contains(path, OLD_TILING_MARKER),
            "new_tiling_marker": file_contains(path, NEW_TILING_MARKER),
        }
    except OSError as exc:
        return {"path": str(path), "error": repr(exc)}


def op_details(torch: Any, qualified_name: str) -> dict[str, Any]:
    namespace_name, op_name = qualified_name.split("::", 1)
    details: dict[str, Any] = {"registered": False}
    try:
        namespace = getattr(torch.ops, namespace_name)
        op = getattr(namespace, op_name)
        details["registered"] = True
        details["schema"] = str(op.default._schema)
    except (AttributeError, RuntimeError) as exc:
        details["error"] = repr(exc)
        return details
    try:
        details["dispatch_table"] = torch._C._dispatch_dump_table(qualified_name)
    except (AttributeError, RuntimeError) as exc:
        details["dispatch_error"] = repr(exc)
    return details


def snapshot(torch: Any, report: dict[str, Any], stage: str) -> None:
    paths = mapped_libraries()
    data = {
        "stage": stage,
        "mapped_libraries": [describe_path(path) for path in paths],
        "global_symbol_owners": global_symbol_owners(),
        "ops": {
            "custom::compressor": op_details(torch, "custom::compressor"),
            **{
                f"_C_ascend::{name}": op_details(torch, f"_C_ascend::{name}")
                for name in SPARSE_OPS
            },
        },
    }
    loaded = getattr(torch.ops, "loaded_libraries", None)
    if isinstance(loaded, set):
        data["torch_ops_loaded_libraries"] = sorted(str(item) for item in loaded)
    report["snapshots"].append(data)

    print(f"\n===== {stage} =====", flush=True)
    print("global ACLNN symbol owners:", flush=True)
    for symbol, owner in data["global_symbol_owners"].items():
        print(f"  {symbol}: {owner}", flush=True)
    if not paths:
        print("mapped custom-op libraries: <none>", flush=True)
    for item in data["mapped_libraries"]:
        print(
            "mapped: {path} old={old_tiling_marker} new={new_tiling_marker} "
            "id={first_4m_sha256}".format(
                path=item.get("path"),
                old_tiling_marker=item.get("old_tiling_marker"),
                new_tiling_marker=item.get("new_tiling_marker"),
                first_4m_sha256=item.get("first_4m_sha256"),
            ),
            flush=True,
        )


def import_custom_module(
    args: argparse.Namespace, torch: Any, report: dict[str, Any]
) -> bool:
    try:
        module = __import__(args.custom_module)
        report["events"].append(
            {
                "event": "import_custom_module",
                "ok": True,
                "module": args.custom_module,
                "module_file": getattr(module, "__file__", None),
            }
        )
        print(f"[PASS] imported {args.custom_module}: {getattr(module, '__file__', None)}")
        ok = True
    except BaseException as exc:
        report["events"].append(
            {
                "event": "import_custom_module",
                "ok": False,
                "module": args.custom_module,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        print(f"[FAIL] import {args.custom_module}: {exc!r}")
        ok = False
    snapshot(torch, report, "after custom module import")
    return ok


def load_binding(binding: Path, torch: Any, report: dict[str, Any]) -> bool:
    try:
        torch.ops.load_library(str(binding))
        report["events"].append(
            {"event": "load_binding", "ok": True, "path": str(binding)}
        )
        print(f"[PASS] torch.ops.load_library({binding})")
        ok = True
    except BaseException as exc:
        report["events"].append(
            {
                "event": "load_binding",
                "ok": False,
                "path": str(binding),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        print(f"[FAIL] load binding: {exc!r}")
        ok = False
    snapshot(torch, report, "after _C_ascend binding load")
    return ok


def call_compressor(torch: Any, device: Any, report: dict[str, Any]) -> bool:
    """Execute a small valid C4A compressor call to trigger ACLNN resolution."""
    try:
        # CANN documents T=0 support, but T=1 makes it less likely that a Python
        # or C++ wrapper returns before resolving aclnnCompressor.
        hidden_size = 1024
        head_dim = 128
        coff = 2
        ratio = 4
        state_width = 2 * coff * head_dim
        x = torch.zeros((1, hidden_size), dtype=torch.bfloat16, device=device)
        wkv = torch.zeros(
            (coff * head_dim, hidden_size), dtype=torch.bfloat16, device=device
        )
        wgate = torch.zeros_like(wkv)
        state_cache = torch.zeros(
            (1, 8, state_width), dtype=torch.float32, device=device
        )
        ape = torch.zeros(
            (ratio, coff * head_dim), dtype=torch.bfloat16, device=device
        )
        norm_weight = torch.ones(hidden_size, dtype=torch.float32, device=device)
        rope_sin = torch.zeros((1, 64), dtype=torch.float32, device=device)
        rope_cos = torch.ones((1, 64), dtype=torch.float32, device=device)
        result = torch.ops.custom.compressor(
            x,
            wkv,
            wgate,
            state_cache,
            ape,
            norm_weight,
            rope_sin=rope_sin,
            rope_cos=rope_cos,
            rope_head_dim=64,
            cmp_ratio=ratio,
            state_block_table=torch.zeros(1, dtype=torch.int32, device=device),
            cu_seqlens=torch.tensor([0, 1], dtype=torch.int32, device=device),
            seqused=torch.ones(1, dtype=torch.int32, device=device),
            start_pos=torch.zeros(1, dtype=torch.int32, device=device),
            coff=coff,
            norm_eps=1e-6,
            rotary_mode=2,
            cache_mode=2,
        )
        torch.npu.synchronize()
        report["events"].append(
            {
                "event": "call_compressor",
                "ok": True,
                "shape": list(result.shape),
                "dtype": str(result.dtype),
            }
        )
        print(f"[PASS] custom.compressor: shape={tuple(result.shape)} dtype={result.dtype}")
        ok = True
    except BaseException as exc:
        report["events"].append(
            {
                "event": "call_compressor",
                "ok": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        print(f"[FAIL] custom.compressor: {exc!r}")
        ok = False
    snapshot(torch, report, "after custom.compressor call")
    return ok


def call_sparse_attention(
    args: argparse.Namespace,
    torch: Any,
    device: Any,
    report: dict[str, Any],
    label: str,
) -> bool:
    try:
        metadata_op = getattr(torch.ops._C_ascend, SPARSE_OPS[0])
        attention_op = getattr(torch.ops._C_ascend, SPARSE_OPS[1])
        for name, op in zip(SPARSE_OPS, (metadata_op, attention_op)):
            print(f"{name} schema: {op.default._schema}")

        batch_size = 1
        num_q_heads = 64
        num_kv_heads = 1
        head_dim = 512
        packed_kv_dim = 640
        page_size = 128
        query_len = args.draft_tokens
        kv_len = args.window_size + query_len
        ori_win_left = kv_len - 1
        index_width = math.ceil(kv_len / 128) * 128
        num_pages = math.ceil(kv_len / page_size)
        cu_seqlens_q = torch.tensor(
            [0, query_len], dtype=torch.int32, device=device
        )
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
        snapshot(torch, report, f"{label}: after sparse metadata")

        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is None:
            raise RuntimeError("torch.float8_e4m3fn is unavailable")
        q = torch.zeros(
            (query_len, num_q_heads, head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        ori_kv = torch.zeros(
            (num_pages, page_size, num_kv_heads, packed_kv_dim),
            dtype=fp8_dtype,
            device=device,
        )
        ori_block_table = torch.arange(
            num_pages, dtype=torch.int32, device=device
        ).view(1, -1)
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
        if tuple(out.shape) != tuple(q.shape):
            raise AssertionError(f"unexpected output shape {tuple(out.shape)}")
        report["events"].append(
            {
                "event": "call_sparse_attention",
                "label": label,
                "ok": True,
                "out_shape": list(out.shape),
                "out_dtype": str(out.dtype),
                "lse_shape": list(softmax_lse.shape),
            }
        )
        print(f"[PASS] {label}: _C_ascend sparse attention")
        ok = True
    except BaseException as exc:
        report["events"].append(
            {
                "event": "call_sparse_attention",
                "label": label,
                "ok": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        print(f"[FAIL] {label}: _C_ascend sparse attention: {exc!r}")
        ok = False
    snapshot(torch, report, f"{label}: after sparse attention attempt")
    return ok


def run_child(args: argparse.Namespace) -> int:
    binding, vendor = resolve_paths(args)
    scenario = args.scenario
    if scenario == "all":
        raise RuntimeError("--child requires one concrete scenario")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "scenario": scenario,
        "pid": os.getpid(),
        "python": sys.version,
        "binding": str(binding),
        "sparse_vendor": str(vendor),
        "environment": {
            "ASCEND_CUSTOM_OPP_PATH": os.getenv("ASCEND_CUSTOM_OPP_PATH"),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH"),
            "LD_DEBUG_OUTPUT": os.getenv("LD_DEBUG_OUTPUT"),
        },
        "events": [],
        "snapshots": [],
    }
    report_path = args.output_dir / f"report-{scenario}.json"
    success = True
    try:
        import torch
        import torch_npu  # noqa: F401 -- registers the NPU dispatch key

        report["torch"] = torch.__version__
        report["torch_npu_file"] = getattr(torch_npu, "__file__", None)
        report["custom_module_spec"] = str(
            __import__("importlib.util").util.find_spec(args.custom_module)
        )
        if not torch.npu.is_available():
            raise RuntimeError("torch_npu imported, but torch.npu.is_available() is false")
        torch.npu.set_device(args.device)
        device = torch.device(f"npu:{args.device}")
        report["device"] = torch.npu.get_device_name(args.device)
        print("PID:", os.getpid())
        print("scenario:", scenario)
        print("device:", report["device"])
        print("ASCEND_CUSTOM_OPP_PATH:", os.getenv("ASCEND_CUSTOM_OPP_PATH"))
        print("LD_LIBRARY_PATH:", os.getenv("LD_LIBRARY_PATH"))
        snapshot(torch, report, "after torch_npu import")

        if scenario == "sparse-only":
            success &= load_binding(binding, torch, report)
            success &= call_sparse_attention(args, torch, device, report, "sparse-only")
        elif scenario == "custom-import-then-sparse":
            success &= import_custom_module(args, torch, report)
            success &= load_binding(binding, torch, report)
            success &= call_sparse_attention(
                args, torch, device, report, "after-custom-import"
            )
        elif scenario == "compressor-then-sparse":
            success &= import_custom_module(args, torch, report)
            success &= call_compressor(torch, device, report)
            success &= load_binding(binding, torch, report)
            success &= call_sparse_attention(args, torch, device, report, "after-compressor")
        elif scenario == "sparse-then-compressor-then-sparse":
            success &= load_binding(binding, torch, report)
            success &= call_sparse_attention(args, torch, device, report, "before-compressor")
            success &= import_custom_module(args, torch, report)
            success &= call_compressor(torch, device, report)
            success &= call_sparse_attention(args, torch, device, report, "after-compressor")
    except BaseException as exc:
        success = False
        report["fatal_error"] = repr(exc)
        report["fatal_traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        report["success"] = bool(success)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("report:", report_path)
    return 0 if success else 1


def main() -> int:
    args = parse_args()
    if args.child:
        return run_child(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
