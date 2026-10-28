"""Build :class:`~esbopfn.core.ProblemSpec` objects from ``configs/problems.yaml``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from esbopfn.config import configs_dir
from esbopfn.core import ProblemSpec
from esbopfn.util.yaml import load_yaml

#: Problem names in the order used by tables and figures.
PROBLEM_NAMES: tuple[str, ...] = ("boiler", "threetank_pi", "pt2_deadtime", "cartpole", "furuta")


def _spec_from_entry(name: str, entry: dict) -> ProblemSpec:
    theta = entry["theta"]
    return ProblemSpec(
        name=name,
        d=len(theta),
        theta_lo=np.array([float(p["lo"]) for p in theta]),
        theta_hi=np.array([float(p["hi"]) for p in theta]),
        T=int(entry["T"]),
        channels=tuple(entry["channels"]),
        rate_channels=tuple(entry.get("rate_channels") or ()),
        can_crash=bool(entry["can_crash"]),
        has_success=bool(entry["has_success"]),
        theta_names=tuple(str(p["name"]) for p in theta),
    )


def load_problem_specs(path: Path | str | None = None) -> dict[str, ProblemSpec]:
    """Load every problem spec, keyed by name, in file order."""
    raw = load_yaml(path if path is not None else configs_dir() / "problems.yaml")
    return {name: _spec_from_entry(name, entry) for name, entry in raw.items()}


def get_spec(name: str, path: Path | str | None = None) -> ProblemSpec:
    """Return the spec of one problem, raising ``KeyError`` with the known names otherwise."""
    specs = load_problem_specs(path)
    if name not in specs:
        raise KeyError(f"unknown problem {name!r}, known: {', '.join(specs)}")
    return specs[name]
