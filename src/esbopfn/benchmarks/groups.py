"""Plot-group specs: which benchmark units a group of thesis figures needs.

A spec is ``configs/benchmarks/<group>.yaml`` with ``kinds`` (the unit kinds, run in the order of
:data:`KIND_ORDER`), shared ``problems`` and ``instances: {count, seed}``, and one section per kind
with its parameters (models or strategies with their params, context sizes, seeds, instances).
Each kind's ``enumerate(cfg)`` reads its own section. ``participants`` declares the model set of
the per-problem normalization ``med_D`` and ``reference_strategies`` the strategy set that
defines ``J_best``, so a figure's normalization never depends on what else is in the database.

``load(name)`` reads a spec, ``units(cfg)`` expands it into ``(kind, key)`` pairs, and
``pfn bench plan|run <group>`` runs them. The plot modules read the same specs.
"""

from __future__ import annotations

from typing import Any

from esbopfn.config import configs_dir
from esbopfn.util.yaml import load_yaml

#: Dependency order of the kinds (a kind may read units of the kinds before it).
KIND_ORDER: tuple[str, ...] = (
    "instances", "fields", "prediction", "prediction_ladder", "prediction_timing",
    "optimization", "optimization_timing", "generation_speed", "landscapes",
    "hardware_optimization", "hardware_prediction", "thesis_series",
)
PROBLEMS: tuple[str, ...] = ("boiler", "threetank_pi", "pt2_deadtime", "cartpole", "furuta")


def kind_rank(kind: str) -> int:
    return KIND_ORDER.index(kind) if kind in KIND_ORDER else len(KIND_ORDER)


def spec_names() -> list[str]:
    """Names of every shipped spec."""
    return sorted(p.stem for p in (configs_dir() / "benchmarks").glob("*.yaml"))


def load(name: str) -> dict[str, Any]:
    path = configs_dir() / "benchmarks" / f"{name}.yaml"
    if not path.exists():
        raise KeyError(f"no benchmark spec {name!r}, known: {', '.join(spec_names())}")
    cfg = load_yaml(path)
    cfg.setdefault("name", name)
    return cfg


def section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    sec = cfg.get(name)
    return dict(sec) if isinstance(sec, dict) else {}


def problems(cfg: dict[str, Any], sec: dict[str, Any] | None = None) -> list[str]:
    return list((sec or {}).get("problems") or cfg.get("problems") or PROBLEMS)


def instances_seed(cfg: dict[str, Any]) -> Any:
    from esbopfn.benchmarks.instances import BENCH_SEED

    return section(cfg, "instances").get("seed", BENCH_SEED)


def units(cfg: dict[str, Any], kinds: list[str] | None = None) -> list[tuple[str, dict[str, Any]]]:
    """``(kind, key)`` pairs of the spec's kinds (restricted to ``kinds``), in dependency order.

    Kinds whose inputs are not complete yet (prediction units need a frozen task table, which
    needs the whole field) contribute nothing until a later plan.
    """
    from esbopfn.benchmarks.kinds import discover_kinds

    registry = discover_kinds()
    wanted = [k for k in cfg.get("kinds", []) if kinds is None or k in kinds]
    out: list[tuple[str, dict[str, Any]]] = []
    for kind in sorted(wanted, key=kind_rank):
        if kind not in registry:
            raise KeyError(f"spec {cfg.get('name')!r} names unknown kind {kind!r}")
        out += [(kind, key) for key in registry[kind].enumerate(cfg)]
    return out


def participants(cfg: dict[str, Any], problem: str, figure: str | None = None) -> list[str]:
    """Resolved model ids of the ``med_D`` participant set (of ``figure`` if the spec declares
    participants per figure)."""
    from esbopfn.benchmarks.kinds.prediction import resolve_model

    spec = cfg.get("participants", [])
    if isinstance(spec, dict):
        spec = spec.get(figure, spec.get("default", []))
    return [resolve_model(m, problem) for m in spec]


def spec_group(name: str):
    """A :class:`PlotGroup` for a spec without a plot module: its needs come from the spec,
    rendering raises :class:`NotImplementedError`."""
    from esbopfn.plots import PlotGroup

    def needs(cfg: dict[str, Any], db: Any) -> list[tuple[str, dict[str, Any]]]:
        return units(cfg or load(name))

    def render(*_: Any) -> None:
        raise NotImplementedError(f"plot group {name!r} has no plot module")

    return PlotGroup(name=name, needs=needs, render=render)
