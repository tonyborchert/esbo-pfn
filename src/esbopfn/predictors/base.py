"""Prediction types, the predictor protocol, the conditioned log-likelihood and the registry.

A prediction is a distribution over the total cost ``J_T`` of one trial with a crash atom at
``+inf`` (thesis, crash-aware output heads). The value part is either a bar distribution
(:class:`BarPrediction`) or a Gaussian (:class:`GaussianPrediction`). Both store the
*unconditional* masses: the value part integrates to ``1 - p_crash``.

Two ways of restricting a prediction to ``J >= j`` exist:

- :meth:`conditioned_above` renormalizes the value mass above ``j`` back to ``1 - p_crash`` and
  keeps ``p_crash`` unchanged. This is the conditioning of the thesis metric (setup, prediction
  metrics) and of the sampled virtual observations.
- :meth:`truncate_below` conditions the whole mixed distribution on the event ``J >= j``. The
  crash atom lies in that event, so ``p_crash`` grows to ``p / (p + S(j))``.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

import numpy as np
from scipy.special import log_ndtr, ndtri
from scipy.stats import truncnorm

from esbopfn.core import ProblemSpec, Trial

#: Default probability and density floor of the conditioned log-likelihood (thesis metric).
LL_FLOOR = 1e-10

#: Variance used to evaluate the density of a degenerate Gaussian exactly at its point mass.
_VAR_FLOOR = 1e-24
_MASS_TOL = 1e-5
_LOG_2PI = math.log(2.0 * math.pi)


def _scalar_or_array(x: np.ndarray, like: Any) -> float | np.ndarray:
    return float(x) if np.ndim(like) == 0 else x


def _check_p_crash(p: float) -> float:
    p = float(p)
    if not 0.0 <= p <= 1.0 or not math.isfinite(p):
        raise ValueError(f"p_crash must lie in [0, 1], got {p}")
    return p


# --------------------------------------------------------------------------------------- bars
@dataclass(frozen=True)
class BarPrediction:
    """Crash-aware bar distribution over ``J_T``.

    ``edges`` has ``N_bin + 1`` strictly increasing entries in raw cost units, ``probs`` holds the
    unconditional mass of each bar and sums to ``1 - p_crash``. Bars are half-open ``(a, b]``
    except the first, which also contains its lower edge. The value density is uniform inside
    each bar.
    """

    edges: np.ndarray
    probs: np.ndarray
    p_crash: float = 0.0

    def __post_init__(self) -> None:
        edges = np.asarray(self.edges, dtype=np.float64).reshape(-1)
        probs = np.asarray(self.probs, dtype=np.float64).reshape(-1)
        p = _check_p_crash(self.p_crash)
        if edges.size < 2 or probs.size != edges.size - 1:
            raise ValueError("need N+1 edges for N probabilities, N >= 1")
        if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
            raise ValueError("edges must be finite and strictly increasing")
        if np.any(probs < 0) or not np.all(np.isfinite(probs)):
            raise ValueError("probs must be finite and non-negative")
        total = float(probs.sum())
        target = 1.0 - p
        if abs(total - target) > _MASS_TOL * max(1.0, target):
            raise ValueError(f"probs sum to {total}, expected 1 - p_crash = {target}")
        if total > 0:
            probs = probs * (target / total)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "probs", probs)
        object.__setattr__(self, "p_crash", p)

    # -- helpers ------------------------------------------------------------------------------
    @property
    def value_mass(self) -> float:
        """Probability of not crashing, ``1 - p_crash``."""
        return 1.0 - self.p_crash

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.edges)

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    def _cum(self) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(self.probs)])

    def _bin_index(self, y: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.edges, y, side="left") - 1
        return np.where(y == self.edges[0], 0, idx)

    def _value_cdf(self, y: np.ndarray) -> np.ndarray:
        """Unconditional value mass at or below ``y``."""
        return np.interp(y, self.edges, self._cum(), left=0.0, right=self._cum()[-1])

    def value_survival(self, y: float | np.ndarray) -> float | np.ndarray:
        """Fraction of the value part above ``y``, ``P(J > y | no crash)``."""
        yy = np.asarray(y, dtype=np.float64)
        if self.value_mass <= 0:
            return _scalar_or_array(np.zeros_like(yy), y)
        cum = self._cum()
        out = np.clip((cum[-1] - self._value_cdf(yy)) / cum[-1], 0.0, 1.0)
        return _scalar_or_array(out, y)

    # -- densities and probabilities ----------------------------------------------------------
    def log_density(self, y: float | np.ndarray) -> float | np.ndarray:
        """Log of the unconditional value density at ``y`` (bar mass over bar width).

        ``-inf`` outside the support, for ``y = +inf`` and inside empty bars.
        """
        yy = np.asarray(y, dtype=np.float64)
        idx = self._bin_index(yy)
        valid = (idx >= 0) & (idx < self.probs.size) & np.isfinite(yy)
        safe = np.clip(idx, 0, self.probs.size - 1)
        dens = self.probs[safe] / self.widths[safe]
        with np.errstate(divide="ignore"):
            out = np.where(valid, np.log(dens), -np.inf)
        return _scalar_or_array(out, y)

    def log_value_density(self, y: float | np.ndarray) -> float | np.ndarray:
        """Log density of the value part conditioned on no crash (integrates to one)."""
        if self.value_mass <= 0:
            return _scalar_or_array(np.full(np.shape(y), -np.inf), y)
        return self.log_density(y) - math.log(self.value_mass)

    def log_prob_crash(self) -> float:
        """``log p_crash`` (``-inf`` when the crash mass is zero)."""
        return math.log(self.p_crash) if self.p_crash > 0 else -math.inf

    def cdf(self, y: float | np.ndarray) -> float | np.ndarray:
        """``P(J_T <= y)`` of the mixed distribution (the crash atom sits at ``+inf``)."""
        yy = np.asarray(y, dtype=np.float64)
        out = np.where(np.isposinf(yy), 1.0, self._value_cdf(np.where(np.isfinite(yy), yy, 0.0)))
        out = np.where(np.isneginf(yy), 0.0, out)
        return _scalar_or_array(out, y)

    def survival(self, y: float | np.ndarray) -> float | np.ndarray:
        """``P(J_T > y)``, the crash mass included."""
        yy = np.asarray(y, dtype=np.float64)
        cum = self._cum()
        finite = np.where(np.isfinite(yy), yy, 0.0)
        out = (cum[-1] - self._value_cdf(finite)) + self.p_crash
        out = np.where(np.isposinf(yy), 0.0, np.where(np.isneginf(yy), 1.0, out))
        return _scalar_or_array(np.clip(out, 0.0, 1.0), y)

    def quantile(self, q: float | np.ndarray) -> float | np.ndarray:
        """Quantile of the mixed distribution, ``+inf`` for ``q`` above ``1 - p_crash``."""
        qq = np.asarray(q, dtype=np.float64)
        if np.any((qq < 0) | (qq > 1)):
            raise ValueError("quantile levels must lie in [0, 1]")
        cum = self._cum()
        out = np.where(qq > cum[-1], np.inf, np.interp(np.minimum(qq, cum[-1]), cum, self.edges))
        return _scalar_or_array(out, q)

    # -- conditioning -------------------------------------------------------------------------
    def _value_part_above(self, j: float, mass: float) -> tuple[np.ndarray, np.ndarray]:
        """Edges and probs of the value part restricted to ``J >= j``, rescaled to ``mass``."""
        edges, probs = self.edges, self.probs
        if j <= edges[0]:
            scale = mass / probs.sum() if probs.sum() > 0 else 0.0
            return edges.copy(), probs * scale
        surviving = float(self._cum()[-1] - self._value_cdf(np.float64(j)))
        if j >= edges[-1] or surviving <= 0.0:
            width = max(abs(j) * 1e-9, 1e-12)
            return np.array([j, j + width]), np.array([mass])
        k = int(np.searchsorted(edges, j, side="right")) - 1
        first = probs[k] * (edges[k + 1] - j) / (edges[k + 1] - edges[k])
        new_edges = np.concatenate([[j], edges[k + 1:]])
        new_probs = np.concatenate([[first], probs[k + 1:]])
        if new_edges[1] - new_edges[0] <= 0:  # j sits on an edge
            new_edges, new_probs = new_edges[1:], new_probs[1:]
        return new_edges, new_probs * (mass / new_probs.sum())

    def conditioned_above(self, j_obs: float) -> BarPrediction:
        """Condition the value part on ``J >= j_obs`` and keep ``p_crash``.

        The value mass above ``j_obs`` is renormalized to ``1 - p_crash``. If no value mass is left
        above ``j_obs``, the value part becomes a point-like bar at ``j_obs``.
        """
        if self.value_mass <= 0:
            return self
        edges, probs = self._value_part_above(float(j_obs), self.value_mass)
        return BarPrediction(edges, probs, self.p_crash)

    def truncate_below(self, lower: float) -> BarPrediction:
        """Condition the mixed distribution on ``J >= lower``, the crash atom included."""
        lower = float(lower)
        if lower <= self.edges[0]:
            return self
        surviving = float(self._cum()[-1] - self._value_cdf(np.float64(lower)))
        total = surviving + self.p_crash
        if total <= 0.0:
            edges, probs = self._value_part_above(lower, 1.0)
            return BarPrediction(edges, probs, 0.0)
        p_new = self.p_crash / total
        if p_new >= 1.0:
            return BarPrediction(self.edges, np.zeros_like(self.probs), 1.0)
        edges, probs = self._value_part_above(lower, 1.0 - p_new)
        return BarPrediction(edges, probs, p_new)

    # -- sampling and moments -----------------------------------------------------------------
    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """``n`` draws, ``+inf`` for crash draws, uniform inside each bar otherwise."""
        u = rng.random(int(n))
        cum = self._cum()
        values = np.interp(np.minimum(u, cum[-1]), cum, self.edges)
        return np.where(u < cum[-1], values, np.inf)

    def mean_var(self) -> tuple[float, float]:
        """Mean and variance of the value part conditioned on no crash (``nan`` if it is empty)."""
        if self.value_mass <= 0:
            return math.nan, math.nan
        w = self.probs / self.probs.sum()
        c, h = self.centers, self.widths
        mean = float(np.sum(w * c))
        second = float(np.sum(w * (c * c + h * h / 12.0)))
        return mean, max(second - mean * mean, 0.0)


# ------------------------------------------------------------------------------------ Gaussian
@dataclass(frozen=True)
class GaussianPrediction:
    """Crash-aware Gaussian over ``J_T``, optionally truncated to ``J >= lower``.

    The value part is ``N(mean, var)`` restricted to ``[lower, inf)`` and renormalized there,
    weighted by ``1 - p_crash``. ``var = 0`` is a point mass at ``max(mean, lower)``.
    """

    mean: float
    var: float
    p_crash: float = 0.0
    lower: float = -math.inf

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.mean)) or not float(self.var) >= 0:
            raise ValueError(f"need a finite mean and var >= 0, got {self.mean}, {self.var}")
        object.__setattr__(self, "mean", float(self.mean))
        object.__setattr__(self, "var", float(self.var))
        object.__setattr__(self, "p_crash", _check_p_crash(self.p_crash))
        object.__setattr__(self, "lower", float(self.lower))

    @property
    def value_mass(self) -> float:
        return 1.0 - self.p_crash

    @property
    def std(self) -> float:
        return math.sqrt(self.var)

    @property
    def degenerate(self) -> bool:
        return self.var <= 0.0

    @property
    def _alpha(self) -> float:
        """Standardized truncation point."""
        if not math.isfinite(self.lower):
            return -math.inf
        return (self.lower - self.mean) / self.std

    @property
    def _log_z(self) -> float:
        """Log of the untruncated normal mass above ``lower``."""
        return float(log_ndtr(-self._alpha)) if math.isfinite(self.lower) else 0.0

    @property
    def _point(self) -> float:
        return max(self.mean, self.lower)

    def value_survival(self, y: float | np.ndarray) -> float | np.ndarray:
        """Fraction of the value part above ``y``, ``P(J > y | no crash)``."""
        yy = np.asarray(y, dtype=np.float64)
        if self.degenerate:
            return _scalar_or_array((yy < self._point).astype(np.float64), y)
        z = (np.maximum(yy, self.lower) - self.mean) / self.std
        out = np.exp(np.minimum(log_ndtr(-z) - self._log_z, 0.0))
        out = np.where(yy < self.lower, 1.0, out)
        return _scalar_or_array(out, y)

    def log_value_density(self, y: float | np.ndarray) -> float | np.ndarray:
        """Log density of the value part conditioned on no crash."""
        yy = np.asarray(y, dtype=np.float64)
        if self.degenerate:
            at = (yy == self._point) & np.isfinite(yy)
            out = np.where(at, -0.5 * (_LOG_2PI + math.log(_VAR_FLOOR)), -np.inf)
            return _scalar_or_array(out, y)
        z = (yy - self.mean) / self.std
        with np.errstate(invalid="ignore"):
            out = -0.5 * (_LOG_2PI + z * z) - math.log(self.std) - self._log_z
        out = np.where((yy < self.lower) | ~np.isfinite(yy), -np.inf, out)
        return _scalar_or_array(out, y)

    def log_density(self, y: float | np.ndarray) -> float | np.ndarray:
        """Log of the unconditional value density, ``log(1 - p_crash) + log f(y)``."""
        if self.value_mass <= 0:
            return _scalar_or_array(np.full(np.shape(y), -np.inf), y)
        return self.log_value_density(y) + math.log(self.value_mass)

    def log_prob_crash(self) -> float:
        return math.log(self.p_crash) if self.p_crash > 0 else -math.inf

    def cdf(self, y: float | np.ndarray) -> float | np.ndarray:
        """``P(J_T <= y)`` of the mixed distribution."""
        yy = np.asarray(y, dtype=np.float64)
        surv = np.asarray(self.value_survival(np.where(np.isfinite(yy), yy, 0.0)))
        out = self.value_mass * (1.0 - surv)
        out = np.where(np.isposinf(yy), 1.0, np.where(np.isneginf(yy), 0.0, out))
        return _scalar_or_array(out, y)

    def survival(self, y: float | np.ndarray) -> float | np.ndarray:
        """``P(J_T > y)``, the crash mass included."""
        yy = np.asarray(y, dtype=np.float64)
        surv = np.asarray(self.value_survival(np.where(np.isfinite(yy), yy, 0.0)))
        out = self.value_mass * surv + self.p_crash
        out = np.where(np.isposinf(yy), 0.0, np.where(np.isneginf(yy), 1.0, out))
        return _scalar_or_array(out, y)

    def _value_ppf(self, v: np.ndarray) -> np.ndarray:
        if self.degenerate:
            return np.full_like(v, self._point)
        if not math.isfinite(self.lower):
            return self.mean + self.std * ndtri(v)
        return truncnorm.ppf(v, self._alpha, np.inf, loc=self.mean, scale=self.std)

    def quantile(self, q: float | np.ndarray) -> float | np.ndarray:
        """Quantile of the mixed distribution, ``+inf`` for ``q`` above ``1 - p_crash``."""
        qq = np.asarray(q, dtype=np.float64)
        if np.any((qq < 0) | (qq > 1)):
            raise ValueError("quantile levels must lie in [0, 1]")
        m = self.value_mass
        if m <= 0:
            return _scalar_or_array(np.full_like(qq, np.inf), q)
        out = np.where(qq > m, np.inf, self._value_ppf(np.minimum(qq / m, 1.0)))
        return _scalar_or_array(out, q)

    def conditioned_above(self, j_obs: float) -> GaussianPrediction:
        """Condition the value part on ``J >= j_obs`` and keep ``p_crash``."""
        return replace(self, lower=max(self.lower, float(j_obs)))

    def truncate_below(self, lower: float) -> GaussianPrediction:
        """Condition the mixed distribution on ``J >= lower``, the crash atom included."""
        lower = float(lower)
        if lower <= self.lower:
            return self
        surv = float(self.value_survival(lower)) * self.value_mass
        total = surv + self.p_crash
        p_new = self.p_crash / total if total > 0 else 0.0
        return replace(self, lower=lower, p_crash=min(p_new, 1.0))

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """``n`` draws, ``+inf`` for crash draws."""
        u = rng.random(int(n))
        m = self.value_mass
        if m <= 0:
            return np.full(int(n), np.inf)
        values = self._value_ppf(np.minimum(u / m, 1.0 - 1e-16))
        return np.where(u < m, values, np.inf)

    def mean_var(self) -> tuple[float, float]:
        """Mean and variance of the (truncated) value part."""
        if self.degenerate:
            return self._point, 0.0
        if not math.isfinite(self.lower):
            return self.mean, self.var
        m, v = truncnorm.stats(self._alpha, np.inf, loc=self.mean, scale=self.std, moments="mv")
        return float(m), max(float(v), 0.0)


Prediction = BarPrediction | GaussianPrediction


# ------------------------------------------------------------------------- log-likelihood
@dataclass(frozen=True)
class LLDetails:
    """One conditioned log-likelihood with the floor events that shaped it."""

    ll: float
    crashed: bool
    clamped_value: bool = False      # the value density term hit the floor
    clamped_crash: bool = False      # log p_crash or log(1 - p_crash) hit the floor
    conditioning_skipped: bool = False  # surviving value mass above j_obs was below the floor


def conditioned_ll_details(
    pred: Prediction, y_true: float, j_obs: float | None = None, floor: float = LL_FLOOR
) -> LLDetails:
    """Conditioned log-likelihood of the thesis metric (setup, prediction metrics), with details.

    A crashed target (``y_true = +inf``) scores ``log p_crash``. Otherwise the score is
    ``log(1 - p_crash) + log f(y_true | J >= j_obs)`` where ``f`` is the value density. The crash
    probability keeps its full predicted mass. Each of the two terms is floored at ``log(floor)``.
    If the value mass surviving above ``j_obs`` is below ``floor``, the conditioning is skipped
    instead of dividing by almost zero.
    """
    log_floor = math.log(floor)
    p = pred.p_crash
    if not math.isfinite(float(y_true)):
        lp = math.log(p) if p > 0 else -math.inf
        return LLDetails(max(lp, log_floor), crashed=True, clamped_crash=lp < log_floor)
    ld = float(pred.log_value_density(float(y_true))) if p < 1 else -math.inf
    skipped = False
    if j_obs is not None and p < 1:
        surviving = float(pred.value_survival(float(j_obs)))
        if surviving >= floor:
            ld -= math.log(surviving)
        else:
            skipped = True
    lnc = math.log1p(-p) if p < 1 else -math.inf
    return LLDetails(
        ll=max(ld, log_floor) + max(lnc, log_floor),
        crashed=False,
        clamped_value=ld < log_floor,
        clamped_crash=lnc < log_floor,
        conditioning_skipped=skipped,
    )


def conditioned_log_likelihood(
    pred: Prediction, y_true: float, j_obs: float | None = None, floor: float = LL_FLOOR
) -> float:
    """Conditioned log-likelihood of ``y_true`` given the cumulative cost ``j_obs`` at the cutoff.

    See :func:`conditioned_ll_details`. Without floor events it equals
    ``pred.conditioned_above(j_obs).log_density(y_true)`` for a finite target and
    ``pred.log_prob_crash()`` for a crash.
    """
    return conditioned_ll_details(pred, y_true, j_obs, floor).ll


# ----------------------------------------------------------------------------- protocol
@runtime_checkable
class Predictor(Protocol):
    """Interface of every predictor (baselines and trained models).

    ``predict`` returns the distribution of ``J_T`` of ``query`` given its observed prefix
    (``query.t_obs`` steps, ``0`` for an unrun controller) and the other trials in ``context``.
    Context-blind predictors ignore ``context``.
    """

    name: str
    supports_unrun: bool
    context_aware: bool

    def predictable_steps(self, spec: ProblemSpec) -> np.ndarray:
        """1-based cutoffs ``t_obs`` at which a prediction is defined."""
        ...

    def predict(self, query: Trial, context: Sequence[Trial]) -> Prediction: ...

    def predict_many(
        self, queries: Sequence[Trial], context: Sequence[Trial]
    ) -> list[Prediction]: ...


class PredictorBase:
    """Convenience base class: defaults for the protocol attributes and a looping
    :meth:`predict_many`. Subclasses override what they batch."""

    name: str = "predictor"
    supports_unrun: bool = False
    context_aware: bool = False

    def predictable_steps(self, spec: ProblemSpec) -> np.ndarray:
        return np.arange(1, spec.T, dtype=np.int64)

    def predict(self, query: Trial, context: Sequence[Trial]) -> Prediction:
        raise NotImplementedError

    def predict_many(
        self, queries: Sequence[Trial], context: Sequence[Trial]
    ) -> list[Prediction]:
        return [self.predict(q, context) for q in queries]

    def _check_query(self, query: Trial) -> None:
        if query.unrun and not self.supports_unrun:
            raise ValueError(f"{self.name} cannot predict an unrun controller (t_obs = 0)")


# ----------------------------------------------------------------------------- registry
#: ``factory(args, spec, **kwargs) -> Predictor`` where ``args`` are the id components after the
#: family prefix, e.g. ``("50",)`` for ``tabpfn/50``.
PredictorFactory = Callable[..., Predictor]

#: Families registered at run time through :func:`register_predictor_family`.
_FAMILIES: dict[str, PredictorFactory] = {}

#: Families resolved lazily on first use, ``prefix -> "module:attribute"`` (entry-point style).
LAZY_FAMILIES: dict[str, str] = {
    "segment-gp": "esbopfn.predictors.segment_gp:make_predictor",
    "tabpfn": "esbopfn.predictors.tabpfn:make_predictor",
    "segment-tabpfn": "esbopfn.predictors.segment_tabpfn:make_predictor",
    "decpfn-nostates": "esbopfn.predictors.decpfn:make_nostates_predictor",
    "decpfn-nodeadtime": "esbopfn.predictors.decpfn:make_nodeadtime_predictor",
    "lcpfn-last": "esbopfn.predictors.lcpfn:make_last_predictor",
}

#: Attribute a family module exposes for convention-based discovery (see :func:`get_predictor`).
FACTORY_ATTRIBUTE = "make_predictor"


def register_predictor_family(
    prefix: str, factory: PredictorFactory | str, *, replace: bool = False
) -> None:
    """Register a predictor family under the id prefix ``prefix``.

    ``factory`` is a callable ``factory(args, spec, **kwargs)`` or a ``"module:attribute"`` string
    that is imported on first use. Registering an existing prefix raises unless ``replace``.
    """
    if not prefix or "/" in prefix:
        raise ValueError(f"invalid family prefix {prefix!r}")
    if not replace and (prefix in _FAMILIES or prefix in LAZY_FAMILIES):
        raise ValueError(f"predictor family {prefix!r} is already registered")
    if isinstance(factory, str):
        _FAMILIES.pop(prefix, None)
        LAZY_FAMILIES[prefix] = factory
    else:
        LAZY_FAMILIES.pop(prefix, None)
        _FAMILIES[prefix] = factory


def predictor_families() -> list[str]:
    """Prefixes known without discovery, sorted."""
    return sorted(set(_FAMILIES) | set(LAZY_FAMILIES))


def _load_target(target: str) -> PredictorFactory:
    module, _, attr = target.partition(":")
    return getattr(importlib.import_module(module), attr or FACTORY_ATTRIBUTE)


def _resolve_family(prefix: str) -> PredictorFactory:
    if prefix in _FAMILIES:
        return _FAMILIES[prefix]
    if prefix in LAZY_FAMILIES:
        factory = _load_target(LAZY_FAMILIES[prefix])
        _FAMILIES[prefix] = factory
        return factory
    module_name = f"esbopfn.predictors.{prefix.replace('-', '_')}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name is not None and module_name.startswith(exc.name):
            known = ", ".join(predictor_families())
            raise KeyError(f"unknown predictor family {prefix!r}, known: {known}") from exc
        raise
    factory = getattr(module, FACTORY_ATTRIBUTE, None)
    if factory is None:
        raise KeyError(f"module {module_name} has no {FACTORY_ATTRIBUTE}(args, spec, **kw)")
    _FAMILIES[prefix] = factory
    return factory


def resolve_spec(problem: ProblemSpec | str | Any) -> ProblemSpec:
    """Accept a :class:`ProblemSpec`, a problem name or an object with a ``SPEC`` attribute."""
    if isinstance(problem, ProblemSpec):
        return problem
    if isinstance(problem, str):
        from esbopfn.problems.registry import get_spec

        return get_spec(problem)
    spec = getattr(problem, "SPEC", None) or getattr(problem, "spec", None)
    if isinstance(spec, ProblemSpec):
        return spec
    raise TypeError(f"cannot derive a ProblemSpec from {problem!r}")


def get_predictor(predictor_id: str, problem: ProblemSpec | str | Any, **kwargs: Any) -> Predictor:
    """Build the predictor named by ``predictor_id`` for ``problem``.

    The id is ``<family>/<args...>``: ``segment-gp``, ``tabpfn/<L>``,
    ``segment-tabpfn/<split|rate>`` and the trained families registered later
    (``decpfn/...``, ``lcpfn/...``, ``xdecpfn/...``, ``ftpfn/...``). A family is found in the
    registry, in :data:`LAZY_FAMILIES`, or by importing ``esbopfn.predictors.<family>`` (dashes
    become underscores) and reading its ``make_predictor``.
    """
    parts = [p for p in predictor_id.strip().strip("/").split("/") if p]
    if not parts:
        raise ValueError("empty predictor id")
    factory = _resolve_family(parts[0])
    return factory(tuple(parts[1:]), resolve_spec(problem), **kwargs)
