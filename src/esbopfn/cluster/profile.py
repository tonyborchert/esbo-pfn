"""Cluster profiles: everything machine specific lives in ``configs/cluster/<name>.yaml``."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from esbopfn.config import configs_dir, profile_name
from esbopfn.util.yaml import load_yaml

JOB_KINDS: tuple[str, ...] = ("gen", "train", "bench", "bench_gpu")

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _closing_brace(value: str, start: int) -> int:
    """Index of the ``}`` closing the ``${`` at ``start`` (nested placeholders allowed), -1 if
    there is none."""
    depth, k = 0, start
    while k < len(value):
        if value.startswith("${", k):
            depth += 1
            k += 2
            continue
        if value[k] == "}":
            depth -= 1
            if depth == 0:
                return k
        k += 1
    return -1


def expand_env(value: str, environ: dict[str, str] | None = None) -> str | None:
    """Expand ``${VAR}`` and ``${VAR:-default}`` placeholders (the default may hold placeholders
    itself, ``${A:-${B}}``), None if one stays unresolved."""
    env = os.environ if environ is None else environ
    out: list[str] = []
    i = 0
    while True:
        j = value.find("${", i)
        if j < 0:
            out.append(value[i:])
            return "".join(out)
        out.append(value[i:j])
        k = _closing_brace(value, j)
        if k < 0:
            return None
        name, sep, default = value[j + 2:k].partition(":-")
        if not _NAME.fullmatch(name):
            return None
        if env.get(name, "") != "":
            out.append(env[name])
        elif sep:
            sub = expand_env(default, env)
            if sub is None:
                return None
            out.append(sub)
        else:
            return None
        i = k + 1


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
    env: tuple[str, ...] = ()
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
        env=tuple(str(v) for v in (raw.get("env") or ())),
        launcher=str(raw.get("launcher") or ""),
        defaults=defaults,
    )
