"""Strategy interface, incumbent helpers and the strategy registry.

A strategy proposes the next controller and may stop a running trial early. The benchmark loop
(:mod:`esbopfn.strategies.loop`) owns budgets, the initial design, simulation, seeds and recording.
Strategies never simulate.

Thesis definitions (background, ESBO): the incumbent ``theta^-_n`` with cost ``J^-_n`` is the best
successful complete trial so far, ``J^+_n`` the largest total cost among the complete trials.
"""

from __future__ import annotations

import importlib
import math
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from esbopfn.core import ProblemSpec, Trial
from esbopfn.predictors.base import resolve_spec


# ------------------------------------------------------------------------------- incumbent
def incumbent_index(history: Sequence[Trial]) -> int | None:
    """Index of the best successful complete trial (first one on ties), ``None`` if none."""
    best, best_j = None, math.inf
    for i, tr in enumerate(history):
        if tr.successful and tr.J_obs < best_j:
            best, best_j = i, tr.J_obs
    return best


def incumbent(history: Sequence[Trial]) -> tuple[np.ndarray | None, float]:
    """``(theta^-_n, J^-_n)``, ``(None, inf)`` without a successful complete trial."""
    i = incumbent_index(history)
    if i is None:
        return None, math.inf
    return np.asarray(history[i].theta, dtype=np.float64), float(history[i].J_obs)


def j_plus(history: Sequence[Trial]) -> float:
    """``J^+_n``, the largest total cost among complete (non-crashed) trials, ``nan`` if none."""
    costs = [tr.J_obs for tr in history if tr.complete]
    return float(max(costs)) if costs else math.nan


def sample_uniform(spec: ProblemSpec, rng: np.random.Generator, n: int | None = None) -> np.ndarray:
    """iid uniform draws from the search box, shape ``(d,)`` or ``(n, d)``."""
    shape = (spec.d,) if n is None else (int(n), spec.d)
    return spec.theta_lo + rng.random(shape) * spec.extent


# -------------------------------------------------------------------------------- strategy
@dataclass(frozen=True)
class StrategySpec:
    """Registry id and the JSON-serializable parameters of a strategy instance."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        from esbopfn.plots.style import DISPLAY_NAME

        return DISPLAY_NAME.get(self.name, self.name)


class Strategy(ABC):
    """Base class of every optimization strategy.

    ``uses_exceed`` tells the loop to apply the ``exceed`` rule. ``decision_steps()`` lists the
    1-based steps at which :meth:`should_stop` is consulted (it is ``False`` at every other step by
    definition). ``records_unarmed`` makes the loop call :meth:`should_stop` also while no stop is
    allowed (initial design, no incumbent), so that predictions can be recorded for later reuse.
    """

    name: ClassVar[str] = "strategy"
    uses_exceed: ClassVar[bool] = True
    records_unarmed: ClassVar[bool] = False

    def __init__(self, spec: ProblemSpec | Any, rng: np.random.Generator, **params: Any) -> None:
        if params:
            raise TypeError(f"{self.name}: unknown parameters {sorted(params)}")
        self.spec: ProblemSpec = resolve_spec(spec)
        self.rng = rng
        self.timing: dict[str, float] = defaultdict(float)
        self.armed = False
        self.J_minus = math.inf
        self.running_index = 0
        self._params: dict[str, Any] = {}

    # -- recorded parameters ------------------------------------------------------------------
    def _set_params(self, **params: Any) -> None:
        self._params.update(params)

    @property
    def strategy_spec(self) -> StrategySpec:
        return StrategySpec(self.name, dict(self._params))

    @contextmanager
    def timed(self, category: str) -> Iterator[None]:
        """Accumulate wall-clock seconds of the block under ``category``."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timing[category] += time.perf_counter() - t0

    # -- interface ------------------------------------------------------------------------------
    @abstractmethod
    def propose(self, history: list[Trial]) -> np.ndarray:
        """Controller parameters ``theta_{n+1}`` (called only after the initial design and once an
        incumbent exists)."""

    def decision_steps(self) -> np.ndarray:
        """Steps at which the strategy's own stop rule is evaluated (none by default)."""
        return np.zeros(0, dtype=np.int64)

    def should_stop(self, running: Trial, history: list[Trial]) -> bool:
        """Own stop rule on the running trial cut at the current step (``exceed`` is the loop's)."""
        return False

    def on_trial_start(self, history: list[Trial], armed: bool) -> None:
        """Called before trial ``len(history)`` runs. ``armed`` is false during the initial design
        and while no incumbent exists."""
        self.running_index = len(history)
        self.armed = bool(armed)
        self.J_minus = incumbent(history)[1]

    def on_trial_end(self, trial: Trial, index: int) -> None:  # noqa: B027 -- optional hook
        """Called once trial ``index`` ended. ``trial.traj`` is the full uncut trajectory, which
        only the oracle strategy reads past ``trial.t_obs``."""

    def incumbent(self, history: Sequence[Trial]) -> tuple[np.ndarray | None, float]:
        return incumbent(history)

    def j_plus(self, history: Sequence[Trial]) -> float:
        return j_plus(history)


# -------------------------------------------------------------------------------- registry
#: ``id -> (module:class, default parameters)``. ``{p}`` in a string default is the problem name.
_REGISTRY: dict[str, tuple[str, dict[str, Any]]] = {
    "esrs": ("esbopfn.strategies.gp_based:ESRS", {}),
    "bo": ("esbopfn.strategies.gp_based:BO", {}),
    "esbo-gp": ("esbopfn.strategies.gp_based:ESBOGP", {}),
    "esbo-oracle": ("esbopfn.strategies.gp_based:ESBOOracle", {}),
    "esbo-t-tabpfn": ("esbopfn.strategies.gp_based:ESBOTTabPFN", {"L": 50}),
    "decpfn": ("esbopfn.strategies.gp_based:DecPFN",
               {"tau": 0.9, "model": "decpfn/{p}/binned/1M"}),
    "decpfn-h": ("esbopfn.strategies.gp_based:DecPFNH",
                 {"tau": 0.9, "model": "decpfn/{p}/binned/1M"}),
    "xdecpfn-pi": ("esbopfn.strategies.pure_loop:PureModelLoop",
                   {"acq": "pi", "tau": 0.99, "model": "xdecpfn/{p}/default/default"}),
    "xdecpfn-ei": ("esbopfn.strategies.pure_loop:PureModelLoop",
                   {"acq": "ei", "tau": 0.99, "model": "xdecpfn/{p}/default/default"}),
    "ftpfn-ei": ("esbopfn.strategies.pure_loop:PureModelLoop",
                 {"acq": "ei", "tau": 0.99, "model": "ftpfn/{p}/default/default"}),
    "ftpfn-pi": ("esbopfn.strategies.pure_loop:PureModelLoop",
                 {"acq": "pi", "tau": 0.99, "model": "ftpfn/{p}/default/default"}),
}

STRATEGY_IDS: tuple[str, ...] = tuple(_REGISTRY)


def _load(target: str) -> Callable[..., Strategy]:
    module, _, attr = target.partition(":")
    return getattr(importlib.import_module(module), attr)


def default_params(strategy_id: str, problem: ProblemSpec | Any) -> dict[str, Any]:
    """Thesis defaults of ``strategy_id`` on ``problem`` (model ids resolved)."""
    if strategy_id not in _REGISTRY:
        raise KeyError(f"unknown strategy {strategy_id!r}, known: {', '.join(STRATEGY_IDS)}")
    spec = resolve_spec(problem)
    return {k: v.format(p=spec.name) if isinstance(v, str) and "{p}" in v else v
            for k, v in _REGISTRY[strategy_id][1].items()}


def make_strategy(strategy_id: str, problem: ProblemSpec | Any, rng: np.random.Generator,
                  **params: Any) -> Strategy:
    """Build ``strategy_id`` for ``problem`` with thesis defaults, overridden by ``params``.

    Common overrides: ``tau``, ``crash_threshold``, ``model``, ``predictor`` (a ready predictor
    object, e.g. a fake in tests), ``cold_refit_growth``, ``acq``, ``L``.
    """
    kwargs = default_params(strategy_id, problem)
    kwargs.update(params)
    cls = _load(_REGISTRY[strategy_id][0])
    strategy = cls(resolve_spec(problem), rng, **kwargs)
    strategy.name = strategy_id
    return strategy
