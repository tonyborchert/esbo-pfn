"""Build :class:`~esbopfn.core.ProblemSpec` and :class:`~esbopfn.problems.prior.PriorSpec` objects
from ``configs/problems.yaml``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from esbopfn.config import configs_dir
from esbopfn.core import ProblemSpec
from esbopfn.problems.prior import BoundedParam, NoiseGroup, Param, PriorSpec, ScaleParam, TiedParam
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


def _param_from_entry(entry: dict) -> Param:
    rule = entry["rule"]
    nominal = entry.get("nominal")
    nominal = None if nominal is None else float(nominal)
    if rule == "scale":
        return ScaleParam(str(entry["name"]), float(entry["nominal"]), float(entry["f_lo"]),
                          float(entry["f_hi"]))
    if rule == "bounded":
        return BoundedParam(str(entry["name"]), float(entry["lo"]), float(entry["hi"]), nominal)
    if rule == "tied":
        return TiedParam(str(entry["name"]), str(entry["source"]), float(entry.get("zeta", 1.5)),
                         nominal)
    raise ValueError(f"unknown prior rule {rule!r}")


def _prior_from_entry(name: str, entry: dict) -> PriorSpec:
    prior = entry["prior"]
    extras = tuple(_param_from_entry(e) for e in prior.get("extras") or ())
    if not all(isinstance(e, BoundedParam) for e in extras):
        raise ValueError(f"{name}: extras must be bounded parameters")
    noise = tuple(NoiseGroup(str(g["name"]), float(g["base"]), tuple(g.get("channels") or ()),
                             str(g.get("kind", "sensor")))
                  for g in entry["noise"])
    return PriorSpec(problem=name, phi=tuple(_param_from_entry(e) for e in prior["phi"]),
                     noise=noise, extras=extras,  # type: ignore[arg-type]
                     uses_phi_hat=bool(prior.get("uses_phi_hat", False)),
                     estimate_sigma_rel=float(prior.get("estimate_sigma_rel", 0.10)))


def load_priors(path: Path | str | None = None) -> dict[str, PriorSpec]:
    """Load the universal prior of every problem, keyed by name, in file order."""
    raw = load_yaml(path if path is not None else configs_dir() / "problems.yaml")
    return {name: _prior_from_entry(name, entry) for name, entry in raw.items()}


def get_prior(name: str, path: Path | str | None = None) -> PriorSpec:
    """Return the universal prior of one problem."""
    priors = load_priors(path)
    if name not in priors:
        raise KeyError(f"unknown problem {name!r}, known: {', '.join(priors)}")
    return priors[name]
