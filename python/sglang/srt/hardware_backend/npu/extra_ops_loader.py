from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_DSPARK_A5_TRACE_ENV = "SGLANG_DSPARK_A5_TRACE_OP_LIBS"
_DSPARK_A5_LIBRARY_MARKERS = (
    "vllm_ascend_c",
    "libcust_opapi",
    "libcust_opmaster",
    "libcust_opsproto",
    "liboptiling",
    "custom_transformer",
)
_DSPARK_A5_OLD_TILING_MARKER = b"oriSparseIndices is not supported now"
_DSPARK_A5_NEW_TILING_MARKER = b"cuSeqLensOriKv is not supported now"
_DSPARK_A5_TRACED_STAGES: set[str] = set()


def _file_contains_marker(path: Path, marker: bytes) -> bool:
    """Search a potentially large shared library without reading it all at once."""
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


def _mapped_dspark_a5_libraries() -> list[Path]:
    maps_path = Path("/proc/self/maps")
    if not maps_path.is_file():
        return []

    paths: set[Path] = set()
    try:
        for line in maps_path.read_text(errors="replace").splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) < 6 or not fields[5].startswith("/"):
                continue
            raw_path = fields[5].removesuffix(" (deleted)")
            if any(marker in raw_path.lower() for marker in _DSPARK_A5_LIBRARY_MARKERS):
                paths.add(Path(raw_path))
    except OSError:
        return []
    return sorted(paths, key=str)


def trace_dspark_a5_op_libraries(stage: str) -> None:
    """Log the DSpark binding/API/tiling libraries mapped by this process.

    Torch's dispatcher identifies an operator by namespace and schema, not by a
    source-file path.  ``/proc/self/maps`` is therefore the reliable way to see
    which ELF objects back the registered binding and the ACLNN host tiling.
    """
    if os.environ.get(_DSPARK_A5_TRACE_ENV, "0").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    if stage in _DSPARK_A5_TRACED_STAGES:
        return
    _DSPARK_A5_TRACED_STAGES.add(stage)

    logger.warning(
        "[DSpark A5 op trace][%s] binding_env=%s custom_opp=%s",
        stage,
        os.environ.get("SGLANG_DSPARK_A5_EXTRA_OPS_SO"),
        os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
    )
    loaded_libraries = getattr(torch.ops, "loaded_libraries", None)
    if isinstance(loaded_libraries, set):
        logger.warning(
            "[DSpark A5 op trace][%s] torch.ops.loaded_libraries=%s",
            stage,
            sorted(loaded_libraries),
        )

    mapped = _mapped_dspark_a5_libraries()
    if not mapped:
        logger.warning(
            "[DSpark A5 op trace][%s] no matching shared libraries in /proc/self/maps",
            stage,
        )
        return
    for path in mapped:
        logger.warning(
            "[DSpark A5 op trace][%s] mapped=%s old_ori_rejection=%s "
            "new_cuseq_rejection=%s",
            stage,
            path,
            _file_contains_marker(path, _DSPARK_A5_OLD_TILING_MARKER),
            _file_contains_marker(path, _DSPARK_A5_NEW_TILING_MARKER),
        )


@dataclass
class OpLibSpec:
    """Configuration for a standalone operator library loaded into ``torch.ops``."""

    name: str  # human-readable id, used in logs/errors
    so_env: str  # env var that points to the standalone .so path
    namespace: str  # torch.ops.<namespace> the operators register into
    required_ops: tuple[str, ...]
    pre_load_imports: tuple[str, ...] = ()  # modules to import before loading


class TorchOpLoader:
    """
    Loader for PyTorch custom operators from shared libraries.

    This class handles the registration and initialization of custom PyTorch
    operators (Ops) from dynamically linked shared object (.so) files. It supports
    environment-based library path discovery, dependency pre-loading, and
    operator existence validation.

    Usage:
        1. Create an OpLibSpec with operator metadata
        2. Instantiate TorchOpLoader with the spec
        3. Call initialize() to load and register the operators

    Example:
        >>> spec = OpLibSpec(
        ...     name="My custom ops",
        ...     so_env="MY_LIB_SO_PATH",
        ...     namespace="_C_my_lib",
        ...     required_ops=("op1", "op2"),
        ...     pre_load_imports=("torch", "other_dep"),
        ... )
        >>> loader = TorchOpLoader(spec)
        >>> lib_path = loader.initialize()
        >>> # Ops are now registered under namespace: _C_my_lib.op1()

    The loader will raise appropriate exceptions if:
        - The shared library cannot be found (via SO_PATH env var or default paths)
        - Pre-load imports fail
        - Required operators are missing after loading
    """

    def __init__(self, spec: OpLibSpec) -> None:
        self._spec = spec
        self._loaded_library: Optional[Path] = None

    def _missing_ops(self) -> list[str]:
        namespace = getattr(torch.ops, self._spec.namespace, None)
        if namespace is None:
            return list(self._spec.required_ops)
        return [op for op in self._spec.required_ops if not hasattr(namespace, op)]

    def registered(self) -> bool:
        """Return whether the required operators are already registered."""
        return not self._missing_ops()

    def _resolve_so_path(self) -> Path:
        explicit = os.environ.get(self._spec.so_env)
        if not explicit:
            raise RuntimeError(
                f"The {self._spec.name} operators are not registered. Set "
                f"{self._spec.so_env} to the standalone .so library path."
            )
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"{self._spec.so_env} points to a missing file: {path}")
        return path

    def _validate_python_abi(self, library_path: Path) -> None:
        abi_match = re.search(r"\.cpython-(\d+)-", library_path.name)
        current_abi = f"{sys.version_info.major}{sys.version_info.minor}"
        if abi_match is not None and abi_match.group(1) != current_abi:
            raise RuntimeError(
                f"{library_path} was built for CPython {abi_match.group(1)}, "
                f"but SGLang is running CPython {current_abi}. Rebuild the "
                "extension with the SGLang Python/Torch/torch-npu environment."
            )

    def initialize(self) -> Optional[Path]:
        """Register the operators before backend execution.

        Idempotent: returns ``None`` if the operators are already registered
        (e.g. by another package). Otherwise loads the standalone .so pointed
        to by ``so_env`` into ``torch.ops`` and validates the required operators.

        Returns the loaded library path when this call loaded it, else ``None``.
        """
        if self.registered():
            return None
        if self._loaded_library is not None:
            missing = self._missing_ops()
            raise RuntimeError(
                f"Loaded {self._loaded_library}, but required "
                f"{self._spec.namespace} operators are missing: {missing}."
            )

        for module in self._spec.pre_load_imports:
            __import__(module)  # noqa: F401  side-effect imports (e.g. torch_npu)

        library_path = self._resolve_so_path()
        self._validate_python_abi(library_path)
        try:
            torch.ops.load_library(str(library_path))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load the {self._spec.name} operator library "
                f"{library_path}. Ensure its dependent CANN/custom-op libraries "
                "are visible through LD_LIBRARY_PATH and the Ascend OPP setup."
            ) from exc

        if self._spec.so_env == "SGLANG_DSPARK_A5_EXTRA_OPS_SO":
            trace_dspark_a5_op_libraries("after torch.ops.load_library")

        missing = self._missing_ops()
        if missing:
            raise RuntimeError(
                f"Loaded {library_path}, but required "
                f"{self._spec.namespace} operators are missing: {missing}."
            )

        self._loaded_library = library_path
        logger.info("Registered %s operators from %s", self._spec.name, library_path)
        return library_path


_DSPARK_A5_SPARSE_ATTN_LOADER = TorchOpLoader(
    OpLibSpec(
        name="DSpark A5 KV-quant sparse-attention",
        so_env="SGLANG_DSPARK_A5_EXTRA_OPS_SO",
        namespace="_C_ascend",
        required_ops=(
            "npu_kv_quant_sparse_attn_sharedkv_metadata",
            "npu_kv_quant_sparse_attn_sharedkv",
        ),
        pre_load_imports=("torch_npu",),
    )
)


def initialize_dspark_a5_sparse_attn_ops() -> Optional[Path]:
    """Load the standalone A5 DSpark sparse-attention extension.

    The external extension is expected to contain both the metadata operator
    and the attention operator and to register them under ``torch.ops._C_ascend``.
    Loading is idempotent, including when another package registered the same
    operators before SGLang initialized its attention backends.
    """
    return _DSPARK_A5_SPARSE_ATTN_LOADER.initialize()
