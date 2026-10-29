"""``pfn doctor``: check the environment (Python, torch device, compilers, BLAS, checkpoints,
C++ binaries)."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from esbopfn.config import paths, repo_root

CPP_BINARIES: tuple[str, ...] = ("furuta_gen", "cartpole_gen", "plants_gen")


def _first_line(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"error: {exc}"
    text = (out.stdout or out.stderr).strip()
    return text.splitlines()[0] if text else ""


def torch_report() -> str:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        return f"not importable ({exc})"
    if torch.cuda.is_available():
        device = f"cuda ({torch.cuda.get_device_name(0)}, {torch.cuda.device_count()} visible)"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return f"{torch.__version__}, device {device}"


def blas_hints() -> list[str]:
    hints: list[str] = []
    try:
        import numpy as np

        blas = np.__config__.CONFIG["Build Dependencies"]["blas"]
        hints.append(f"numpy BLAS: {blas.get('name')} {blas.get('version', '')}".rstrip())
    except (AttributeError, KeyError, TypeError):
        hints.append("numpy BLAS: unknown")
    if Path("/System/Library/Frameworks/Accelerate.framework").exists():
        hints.append("Accelerate framework: present (cmake -DBLAS=accelerate)")
    if shutil.which("pkg-config"):
        if subprocess.run(["pkg-config", "--exists", "openblas"], check=False).returncode == 0:
            hints.append("OpenBLAS via pkg-config: present (cmake -DBLAS=openblas)")
    if os.environ.get("MKLROOT"):
        hints.append(f"MKLROOT={os.environ['MKLROOT']} (cmake -DBLAS=mkl)")
    return hints


def find_cpp_binaries() -> dict[str, Path | None]:
    """Location of each generator binary in ``$PFN_HOME/bin`` or the checkout's build dirs."""
    dirs = [paths().root / "bin"]
    root = repo_root()
    if root is not None:
        dirs += sorted((root / "cpp").glob("build*")) + [root / "build"]
    found: dict[str, Path | None] = {}
    for name in CPP_BINARIES:
        hit = next((d / name for d in dirs if (d / name).is_file()), None)
        found[name] = hit
    return found


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("doctor", help="check the environment")
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    from esbopfn.predictors.checkpoints import tabpfn_status

    print(f"{'python':<14}{sys.version.split()[0]} ({platform.platform()})")
    print(f"{'torch':<14}{torch_report()}")
    for tool, flag in (("cc", "--version"), ("c++", "--version"), ("cmake", "--version")):
        exe = shutil.which(tool)
        print(f"{tool:<14}{_first_line([exe, flag]) if exe else 'missing'}")
    for hint in blas_hints():
        print(f"{'blas':<14}{hint}")
    st = tabpfn_status()
    detail = f"{st.path.name} ({st.size / 1e6:.1f} MB)" if st.present else "missing"
    print(f"{'tabpfn':<14}{st.path.parent}: {detail}")
    for name, path in find_cpp_binaries().items():
        print(f"{name:<14}{path if path else 'not built'}")
    return 0
