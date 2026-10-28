"""Cluster profiles: everything machine specific lives in ``configs/cluster/<name>.yaml``."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from esbopfn.config import configs_dir, profile_name
from esbopfn.util.yaml import load_yaml

JOB_KINDS: tuple[str, ...] = ("gen", "train", "bench", "bench_gpu")

_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: str, environ: dict[str, str] | None = None) -> str | None:
    """Expand ``${VAR}`` and ``${VAR:-default}`` placeholders, None if one stays unresolved."""
    env = os.environ if environ is None else environ

    def sub(m: re.Match[str]) -> str:
        name, default = m.group(1), m.group(2)
        if name in env and env[name] != "":
            return env[name]
        return default if default is not None else m.group(0)

    expanded = _PLACEHOLDER.sub(sub, value)
    if "${" in expanded:
        expanded = _PLACEHOLDER.sub(sub, expanded)
    return None if "${" in expanded else expanded


@dataclass(frozen=True)
class Resources:
    """Resources of one job."""

    cpus: int
    mem: str
    time: str
    gpus: int = 0


@dataclass(frozen=True)
class Profile:
    """A cluster profile. ``scheduler`` is ``slurm`` or ``none``."""

    name: str
    scheduler: str
    root: str | None = None
    accounts: dict[str, str] = field(default_factory=dict)
    partitions: dict[str, str] = field(default_factory=dict)
    modules: tuple[str, ...] = ()
    modules_cpp: tuple[str, ...] = ()
    pre: str = ""
    pre_gpu: str = ""
    launcher: str = "uv run --no-sync"
    defaults: dict[str, Resources] = field(default_factory=dict)

    def resources(self, kind: str) -> Resources:
        """Default resources of a job kind."""
        if kind not in self.defaults:
            raise KeyError(f"profile {self.name!r} has no defaults for kind {kind!r}")
        return self.defaults[kind]

    def resolved_root(self, environ: dict[str, str] | None = None) -> Path | None:
        """``root`` with environment variables expanded, or None if unset or unresolvable."""
        if not self.root:
            return None
        expanded = expand_env(self.root, environ)
        return None if expanded is None else Path(expanded).expanduser()

    def resolved_account(self, part: str, environ: dict[str, str] | None = None) -> str | None:
        """The account of resource class ``part`` with placeholders expanded, None if unset."""
        raw = self.accounts.get(part)
        if not raw:
            return None
        expanded = expand_env(raw, environ)
        if expanded is None:
            raise RuntimeError(
                f"the {part} Slurm account of profile {self.name!r} is {raw!r}, set the "
                "environment variable it names (see docs/CLUSTER.md)"
            )
        return expanded


def profile_path(name: str) -> Path:
    return configs_dir() / "cluster" / f"{name}.yaml"


def available_profiles() -> list[str]:
    """Names of the shipped profiles."""
    return sorted(p.stem for p in (configs_dir() / "cluster").glob("*.yaml"))


def load_profile(name: str | None = None, path: Path | str | None = None) -> Profile:
    """Load a profile by name (default ``$PFN_PROFILE`` or ``local``) or from an explicit path."""
    if path is None:
        name = name or profile_name()
        path = profile_path(name)
        if not path.exists():
            raise FileNotFoundError(
                f"no cluster profile {name!r}, available: {', '.join(available_profiles())}"
            )
    raw = load_yaml(path)
    scheduler = str(raw.get("scheduler", "none"))
    if scheduler not in ("slurm", "none"):
        raise ValueError(f"unknown scheduler {scheduler!r} in {path}")
    defaults = {
        kind: Resources(
            cpus=int(res.get("cpus", 1)),
            mem=str(res.get("mem", "4G")),
            time=str(res.get("time", "01:00:00")),
            gpus=int(res.get("gpus", 0)),
        )
        for kind, res in (raw.get("defaults") or {}).items()
    }
    return Profile(
        name=str(raw.get("name", Path(path).stem)),
        scheduler=scheduler,
        root=raw.get("root"),
        accounts=dict(raw.get("accounts") or {}),
        partitions=dict(raw.get("partitions") or {}),
        modules=tuple(raw.get("modules") or ()),
        modules_cpp=tuple(raw.get("modules_cpp") or ()),
        pre=str(raw.get("pre") or ""),
        pre_gpu=str(raw.get("pre_gpu") or ""),
        launcher=str(raw.get("launcher") or ""),
        defaults=defaults,
    )
