"""Resolution of the output root ``PFN_HOME``, the cluster profile name and the config directory.

Two environment variables configure this project. ``PFN_HOME`` is the output root (default
``./out`` relative to the current working directory) and ``PFN_PROFILE`` names the cluster profile
(default ``local``). Both are read when a function is called, never at import time. The cluster
profiles additionally read the Slurm accounts (``PFN_SLURM_ACCOUNT_CPU``, ``PFN_SLURM_ACCOUNT_GPU``)
and their ``${HPCWORK}``/``${PROJECT}`` placeholders (:mod:`esbopfn.cluster.profile`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_HOME = "PFN_HOME"
ENV_PROFILE = "PFN_PROFILE"
DEFAULT_PROFILE = "local"

#: Subdirectories of ``PFN_HOME`` (see the output tree in the README).
SUBDIRS: tuple[str, ...] = (
    "datasets",
    "models",
    "hpo",
    "checkpoints",
    "benchmarks",
    "figures",
    "jobs",
    "logs",
    "bin",
)


def pfn_home(environ: dict[str, str] | None = None) -> Path:
    """Return the absolute output root: ``$PFN_HOME`` if set and non-empty, else ``./out``."""
    env = os.environ if environ is None else environ
    raw = env.get(ENV_HOME, "").strip()
    home = Path(raw).expanduser() if raw else Path.cwd() / "out"
    return home.resolve()


def profile_name(environ: dict[str, str] | None = None) -> str:
    """Return the cluster profile name: ``$PFN_PROFILE`` if set and non-empty, else ``local``."""
    env = os.environ if environ is None else environ
    return env.get(ENV_PROFILE, "").strip() or DEFAULT_PROFILE


def configs_dir() -> Path:
    """Return the directory holding the shipped YAML configs (``esbopfn/configs``)."""
    return Path(__file__).resolve().parent / "configs"


@dataclass(frozen=True)
class Paths:
    """The output tree below one root. Every attribute access creates its directory on demand."""

    root: Path

    def sub(self, name: str) -> Path:
        """Return ``root/name`` and create it if missing."""
        path = self.root / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def datasets(self) -> Path:
        return self.sub("datasets")

    @property
    def models(self) -> Path:
        return self.sub("models")

    @property
    def hpo(self) -> Path:
        return self.sub("hpo")

    @property
    def checkpoints(self) -> Path:
        return self.sub("checkpoints")

    @property
    def benchmarks(self) -> Path:
        return self.sub("benchmarks")

    @property
    def figures(self) -> Path:
        return self.sub("figures")

    @property
    def jobs(self) -> Path:
        return self.sub("jobs")

    @property
    def logs(self) -> Path:
        return self.sub("logs")

    @property
    def bin(self) -> Path:
        return self.sub("bin")

    @property
    def db_file(self) -> Path:
        """The benchmark database file (its directory is created, the file is not)."""
        return self.benchmarks / "db.sqlite"

    @property
    def tabpfn_checkpoints(self) -> Path:
        """Directory for the TabPFN regressor weights (not created, it is a lookup location)."""
        return self.root / "checkpoints" / "tabpfn"

    def as_dict(self) -> dict[str, Path]:
        """Map every subdirectory name to its path without creating anything."""
        return {name: self.root / name for name in SUBDIRS}


def paths(home: Path | str | None = None) -> Paths:
    """Return the output tree for ``home`` (default :func:`pfn_home`). Nothing is created yet."""
    root = Path(home).expanduser().resolve() if home is not None else pfn_home()
    return Paths(root=root)


def repo_root() -> Path | None:
    """Root of the source checkout (the directory holding ``pyproject.toml``), or None when the
    package is installed outside a checkout."""
    here = Path(__file__).resolve().parent
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "esbopfn").is_dir():
            return parent
    return None
