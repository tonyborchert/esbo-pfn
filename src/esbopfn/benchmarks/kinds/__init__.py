"""Benchmark kinds. Each module here exposes ``KIND = BenchmarkKind(...)``."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Key = dict[str, Any]


@dataclass(frozen=True)
class BenchmarkKind:
    """An atomic unit type of the benchmark database.

    ``enumerate(spec)`` expands a plot-group spec into unit keys. ``run(key, blob_dir)`` computes
    one unit and writes its results into ``blob_dir``.
    """

    name: str
    enumerate: Callable[[dict[str, Any]], list[Key]]
    run: Callable[[Key, Path], None]


def discover_kinds() -> dict[str, BenchmarkKind]:
    """Import every module of this package and collect its ``KIND``."""
    kinds: dict[str, BenchmarkKind] = {}
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        kind = getattr(module, "KIND", None)
        if not isinstance(kind, BenchmarkKind):
            raise RuntimeError(f"{module.__name__} does not expose KIND = BenchmarkKind(...)")
        if kind.name in kinds:
            raise RuntimeError(f"duplicate benchmark kind {kind.name!r}")
        kinds[kind.name] = kind
    return kinds
