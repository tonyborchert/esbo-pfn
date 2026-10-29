"""Plot groups. Each module here (other than ``style``) exposes ``GROUP = PlotGroup(...)``.

A group's config is read from ``configs/benchmarks/<group>.yaml`` (empty if the file is absent).
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from esbopfn.config import configs_dir
from esbopfn.util.yaml import load_yaml

if TYPE_CHECKING:
    from esbopfn.benchmarks.db import Db

#: Modules of this package that are helpers, not plot groups.
NON_GROUP_MODULES = frozenset({"style", "common", "examples"})


@dataclass(frozen=True)
class PlotGroup:
    """A set of figures and the benchmark units they need.

    ``needs(cfg, db)`` returns ``(kind, key)`` pairs. ``render(cfg, db, out_dir, fmt, theme)``
    writes the figures into ``out_dir``.
    """

    name: str
    needs: Callable[[dict[str, Any], Db], list[tuple[str, dict[str, Any]]]]
    render: Callable[[dict[str, Any], Db, Path, str, str], None]


def discover_groups() -> dict[str, PlotGroup]:
    """Import every group module of this package and collect its ``GROUP``."""
    groups: dict[str, PlotGroup] = {}
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_") or info.name in NON_GROUP_MODULES:
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        group = getattr(module, "GROUP", None)
        if not isinstance(group, PlotGroup):
            raise RuntimeError(f"{module.__name__} does not expose GROUP = PlotGroup(...)")
        if group.name in groups:
            raise RuntimeError(f"duplicate plot group {group.name!r}")
        groups[group.name] = group
    return groups


def group_config(name: str) -> dict[str, Any]:
    """The YAML config of a group, or an empty dict."""
    path = configs_dir() / "benchmarks" / f"{name}.yaml"
    return load_yaml(path) if path.exists() else {}
