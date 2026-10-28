"""The rig registry ``configs/rigs.yaml`` (open: a new rig configuration is a new entry)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from esbopfn.config import configs_dir


@dataclass(frozen=True)
class Rig:
    name: str
    description: str
    mass_note: str = "none"
    reset_reference_method: str = "decpfn-h"
    j_best: str = "best_on_rig"
    thesis_tree: str | None = None


@dataclass(frozen=True)
class FieldDefaults:
    problem: str = "furuta"
    methods: tuple[str, ...] = ("esrs", "bo", "esbo-gp", "decpfn", "decpfn-h", "xdecpfn-pi")
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    n_max: int = 180
    t_budget: int = 36000
    method_params: dict[str, dict[str, Any]] = field(default_factory=dict)


def rigs_path() -> Path:
    return configs_dir() / "rigs.yaml"


def _load(path: Path | None) -> dict[str, Any]:
    return yaml.safe_load(Path(path or rigs_path()).read_text()) or {}


def load_rigs(path: Path | str | None = None) -> dict[str, Rig]:
    """All rig configurations by name."""
    data = _load(Path(path) if path else None)
    out = {}
    for name, entry in (data.get("rigs") or {}).items():
        entry = dict(entry or {})
        if "description" not in entry:
            raise ValueError(f"rig {name!r} needs a description")
        known = set(Rig.__dataclass_fields__) - {"name"}
        unknown = set(entry) - known
        if unknown:
            raise ValueError(f"rig {name!r}: unknown fields {sorted(unknown)}")
        out[name] = Rig(name=name, **entry)
    return out


def get_rig(name: str, path: Path | str | None = None) -> Rig:
    rigs = load_rigs(path)
    if name not in rigs:
        raise KeyError(f"unknown rig {name!r}, known: {', '.join(sorted(rigs))} "
                       f"(add a new entry to {rigs_path()})")
    return rigs[name]


def field_defaults(path: Path | str | None = None) -> FieldDefaults:
    """Methods, seeds, budgets and method parameters of the rig field."""
    entry = dict(_load(Path(path) if path else None).get("field") or {})
    if "methods" in entry:
        entry["methods"] = tuple(entry["methods"])
    if "seeds" in entry:
        entry["seeds"] = tuple(int(s) for s in entry["seeds"])
    entry["method_params"] = {k: dict(v or {}) for k, v in
                              (entry.get("method_params") or {}).items()}
    return FieldDefaults(**entry)
