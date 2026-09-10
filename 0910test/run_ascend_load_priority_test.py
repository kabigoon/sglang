#!/usr/bin/env python3
"""Run the collision matrix with a temporary CANN vendors/config.ini.

Only the small config-file install/restore operations use sudo.  The actual
diagnostic remains in the invoking user's Python environment.  The original
config is restored in a finally block, including when a scenario fails.

The default intentionally tests the colleague's proposal literally:

    load_priority=<absolute path to vllm-ascend custom_transformer>

Use --priority-style vendor-name for a documented-syntax comparison run:

    load_priority=custom_transformer,customize
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = Path(
    "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/config.ini"
)
DEFAULT_SYSTEM_TRANSFORMER = Path(
    "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer"
)
DEFAULT_CUSTOMIZE = Path(
    "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Temporarily set CANN load_priority, run the isolated operator "
            "collision matrix, and restore config.ini"
        )
    )
    parser.add_argument(
        "--vllm-root",
        type=Path,
        default=Path("/home/a00821909/vllm-ascend"),
    )
    parser.add_argument("--sparse-vendor", type=Path)
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--priority-style",
        choices=("path", "vendor-name"),
        default="path",
        help=(
            "path tests the proposed absolute-path value; vendor-name is the "
            "documented comparator (default: path)"
        ),
    )
    parser.add_argument(
        "--priority-value",
        help="Exact load_priority value; overrides --priority-style",
    )
    parser.add_argument(
        "--system-transformer",
        type=Path,
        default=DEFAULT_SYSTEM_TRANSFORMER,
    )
    parser.add_argument("--customize", type=Path, default=DEFAULT_CUSTOMIZE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "ascend-op-collision-results-config-path-priority",
    )
    parser.add_argument(
        "--diagnostic",
        type=Path,
        default=HERE / "diagnose_ascend_op_collision.py",
    )
    parser.add_argument(
        "--scenario",
        choices=(
            "all",
            "sparse-only",
            "custom-import-then-sparse",
            "compressor-then-sparse",
            "sparse-then-compressor-then-sparse",
        ),
        default="all",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--draft-tokens", type=int, default=5)
    parser.add_argument("--custom-module", default="custom_ops")
    parser.add_argument("--ld-debug", action="store_true")
    return parser.parse_args()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_metadata(path: Path, data: bytes | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "realpath": str(path.resolve(strict=False)),
        "exists": data is not None,
    }
    if data is not None:
        info = path.stat()
        result.update(
            {
                "size": len(data),
                "sha256": sha256(data),
                "mode": oct(stat.S_IMODE(info.st_mode)),
                "uid": info.st_uid,
                "gid": info.st_gid,
            }
        )
    return result


def replace_load_priority(original: bytes | None, priority: str) -> bytes:
    text = (original or b"").decode("utf-8", errors="surrogateescape")
    lines = text.splitlines(keepends=True)
    output: list[str] = []
    replaced = False
    for line in lines:
        body = line.rstrip("\r\n")
        ending = line[len(body) :]
        stripped = body.strip()
        key, separator, _value = stripped.partition("=")
        is_priority = (
            bool(separator)
            and not stripped.startswith(("#", ";"))
            and key.strip() == "load_priority"
        )
        if not is_priority:
            output.append(line)
        elif not replaced:
            output.append(f"load_priority={priority}{ending or chr(10)}")
            replaced = True
        # Drop duplicate active load_priority keys for an unambiguous test.
    if not replaced:
        if output and not output[-1].endswith(("\n", "\r")):
            output[-1] += "\n"
        output.append(f"load_priority={priority}\n")
    return "".join(output).encode("utf-8", errors="surrogateescape")


def privileged(command: list[str]) -> None:
    if os.geteuid() != 0:
        sudo = shutil.which("sudo")
        if sudo is None:
            raise RuntimeError("sudo is required to update the CANN config")
        command = [sudo, *command]
    print("config command:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def install_file(source: Path, destination: Path, mode: int) -> None:
    privileged(["install", "-m", f"{mode:o}", str(source), str(destination)])


def main() -> int:
    args = parse_args()
    vllm_root = args.vllm_root.expanduser().resolve()
    sparse_vendor = (
        args.sparse_vendor.expanduser().resolve()
        if args.sparse_vendor
        else vllm_root / "vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
    )
    if not sparse_vendor.is_dir():
        raise RuntimeError(f"Sparse vendor is missing: {sparse_vendor}")
    diagnostic = args.diagnostic.expanduser().resolve()
    if not diagnostic.is_file():
        raise RuntimeError(f"Diagnostic script is missing: {diagnostic}")

    requested_config = args.config_path.expanduser()
    # Preserve a vendor-tree symlink such as ascend-toolkit/latest by writing to
    # its resolved config target instead of replacing a config.ini symlink.
    effective_config = requested_config.resolve(strict=False)
    if not effective_config.parent.is_dir():
        raise RuntimeError(f"Config parent is missing: {effective_config.parent}")

    original_exists = effective_config.exists()
    if original_exists and not effective_config.is_file():
        raise RuntimeError(f"Config path is not a regular file: {effective_config}")
    original = effective_config.read_bytes() if original_exists else None
    original_info = effective_config.stat() if original_exists else None
    original_mode = stat.S_IMODE(original_info.st_mode) if original_info else 0o644

    if args.priority_value is not None:
        priority = args.priority_value
    elif args.priority_style == "path":
        priority = str(sparse_vendor)
    else:
        priority = "custom_transformer,customize"
    if "\n" in priority or "\r" in priority:
        raise RuntimeError("load_priority must be a single line")

    active = replace_load_priority(original, priority)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_path = output_dir / "config.ini.before"
    active_path = output_dir / "config.ini.active"
    if original is not None:
        backup_path.write_bytes(original)
    active_path.write_bytes(active)

    experiment_path = output_dir / "config-priority-experiment.json"
    experiment: dict[str, Any] = {
        "started_at_unix": time.time(),
        "priority_style": args.priority_style,
        "load_priority": priority,
        "requested_config": str(requested_config),
        "effective_config": str(effective_config),
        "original": file_metadata(effective_config, original),
        "active_sha256": sha256(active),
        "restored": False,
    }
    experiment_path.write_text(
        json.dumps(experiment, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    diagnostic_code = 1
    installed = False
    restore_error: BaseException | None = None
    try:
        install_file(active_path, effective_config, original_mode)
        installed = True
        if original_info is not None:
            privileged(
                [
                    "chown",
                    f"{original_info.st_uid}:{original_info.st_gid}",
                    str(effective_config),
                ]
            )
        live = effective_config.read_bytes()
        if live != active:
            raise RuntimeError(
                "Installed config does not match config.ini.active: "
                f"expected {sha256(active)}, got {sha256(live)}"
            )

        command = [
            sys.executable,
            str(diagnostic),
            "--vllm-root",
            str(vllm_root),
            "--sparse-vendor",
            str(sparse_vendor),
            "--scenario",
            args.scenario,
            "--device",
            str(args.device),
            "--window-size",
            str(args.window_size),
            "--draft-tokens",
            str(args.draft_tokens),
            "--custom-module",
            args.custom_module,
            "--output-dir",
            str(output_dir),
            "--opp-mode",
            "only",
            "--vendor-config",
            str(requested_config),
            "--expected-load-priority",
            priority,
            "--system-vendor",
            str(args.system_transformer.expanduser().resolve()),
            "--system-vendor",
            str(args.customize.expanduser().resolve()),
        ]
        if args.binding:
            command.extend(("--binding", str(args.binding.expanduser().resolve())))
        if args.ld_debug:
            command.append("--ld-debug")
        print("diagnostic command:", " ".join(command), flush=True)
        diagnostic_code = subprocess.run(command, check=False).returncode
        experiment["diagnostic_returncode"] = diagnostic_code
    finally:
        try:
            if installed:
                if original is not None:
                    install_file(backup_path, effective_config, original_mode)
                    assert original_info is not None
                    privileged(
                        [
                            "chown",
                            f"{original_info.st_uid}:{original_info.st_gid}",
                            str(effective_config),
                        ]
                    )
                    restored = effective_config.read_bytes() == original
                else:
                    privileged(["rm", "-f", "--", str(effective_config)])
                    restored = not effective_config.exists()
                if not restored:
                    raise RuntimeError("config.ini restoration verification failed")
                experiment["restored"] = True
        except BaseException as exc:
            restore_error = exc
            experiment["restore_error"] = repr(exc)
        finally:
            experiment["finished_at_unix"] = time.time()
            experiment_path.write_text(
                json.dumps(experiment, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            if restore_error is not None:
                raise RuntimeError(
                    f"CRITICAL: failed to restore {effective_config}: "
                    f"{restore_error!r}; backup is {backup_path}"
                ) from restore_error

    print("experiment report:", experiment_path)
    print("config restored:", experiment["restored"])
    return diagnostic_code


if __name__ == "__main__":
    raise SystemExit(main())
