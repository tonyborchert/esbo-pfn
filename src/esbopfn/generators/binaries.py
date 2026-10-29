"""Locating, building and calling the C++ generator binaries."""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from esbopfn.config import paths, repo_root

log = logging.getLogger(__name__)

#: Problem -> binary serving it.
BINARY_OF: dict[str, str] = {
    "boiler": "plants_gen",
    "threetank_pi": "plants_gen",
    "pt2_deadtime": "plants_gen",
    "cartpole": "cartpole_gen",
    "furuta": "furuta_gen",
}
BLAS_CHOICES = ("accelerate", "openblas", "mkl")


class BinaryMissing(FileNotFoundError):
    """Raised when a generator binary has not been built."""

    def __init__(self, name: str):
        super().__init__(f"{name} not found in $PFN_HOME/bin or cpp/build*, run `pfn data build`")
        self.name = name


def find_binary(problem: str, home: Path | str | None = None) -> Path:
    """Path of the binary for ``problem`` (``$PFN_HOME/bin`` first, then ``cpp/build*``)."""
    if problem not in BINARY_OF:
        raise KeyError(f"unknown problem {problem!r}, known: {', '.join(BINARY_OF)}")
    name = BINARY_OF[problem]
    dirs = [paths(home).root / "bin"]
    root = repo_root()
    if root is not None:
        dirs += sorted((root / "cpp").glob("build*"))
    for d in dirs:
        if (d / name).is_file() and os.access(d / name, os.X_OK):
            return d / name
    raise BinaryMissing(name)


def command(problem: str, sub: str, *args: str, home: Path | str | None = None) -> list[str]:
    """Argument vector ``[binary, sub, --problem P, *args]``."""
    return [str(find_binary(problem, home)), sub, "--problem", problem, *args]


def run(problem: str, sub: str, *args: str, home: Path | str | None = None,
        capture: bool = True, timeout: float | None = None) -> str:
    """Run a subcommand. Returns stdout when ``capture``, raises on a non-zero exit."""
    cmd = command(problem, sub, *args, home=home)
    log.debug("running %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout, check=False)
    if res.returncode != 0:
        err = (res.stderr or "").strip() if capture else ""
        raise RuntimeError(f"{' '.join(cmd[:2])} failed ({res.returncode}): {err}")
    return res.stdout if capture else ""


def run_json(problem: str, sub: str, *args: str, home: Path | str | None = None,
             timeout: float | None = None) -> Any:
    return json.loads(run(problem, sub, *args, home=home, timeout=timeout))


def version(problem: str, home: Path | str | None = None) -> str:
    res = subprocess.run([str(find_binary(problem, home)), "version"], capture_output=True,
                         text=True, check=True)
    return res.stdout.strip()


def info(problem: str, home: Path | str | None = None) -> dict[str, Any]:
    """The binary's description of the problem (names, layout sizes, nominal instance)."""
    return run_json(problem, "info", home=home)


def detect_blas() -> str:
    """Accelerate on macOS, OpenBLAS otherwise (MKL is chosen explicitly with ``--blas mkl``)."""
    if platform.system() == "Darwin":
        return "accelerate"
    return "openblas"


def cpp_source_dir() -> Path:
    root = repo_root()
    if root is None or not (root / "cpp" / "CMakeLists.txt").is_file():
        raise FileNotFoundError("the cpp/ sources are not available (not running from a checkout)")
    return root / "cpp"


def build(blas: str | None = None, home: Path | str | None = None, build_dir: Path | None = None,
          jobs: int | None = None, native: bool = False, extra: Sequence[str] = ()) -> Path:
    """Configure, build and install the three binaries into ``$PFN_HOME/bin``.

    Returns the install bin directory.
    """
    if shutil.which("cmake") is None:
        raise FileNotFoundError("cmake not found on PATH")
    blas = blas or detect_blas()
    if blas not in BLAS_CHOICES:
        raise ValueError(f"--blas must be one of {', '.join(BLAS_CHOICES)}")
    src = cpp_source_dir()
    bdir = Path(build_dir) if build_dir is not None else src / f"build-{blas}"
    prefix = paths(home).root
    jobs = jobs or max(1, os.cpu_count() or 1)
    steps = [
        ["cmake", "-S", str(src), "-B", str(bdir), f"-DBLAS={blas}",
         f"-DNATIVE={'ON' if native else 'OFF'}", "-DCMAKE_BUILD_TYPE=Release", *extra],
        ["cmake", "--build", str(bdir), "-j", str(jobs)],
        ["cmake", "--install", str(bdir), "--prefix", str(prefix)],
    ]
    for cmd in steps:
        log.info("%s", " ".join(cmd))
        subprocess.run(cmd, check=True)
    return prefix / "bin"
