"""The universal prior over system instances, controllers and trajectories.

Implements the recipe of the thesis (setup, domain-randomized prior instantiation, and method,
priors over single trajectories and over trajectory sets):

* physical parameters ``p`` are classed as Scale (log-uniform from ``p0 / f_lo`` to
  ``p0 * f_hi``), Bounded (uniform on a physical domain) or Tied (``p_src * LogUniform(1/zeta,
  zeta)``),
* sensor noise ``sigma_i = base_i * S * eta_i`` with one level ``S ~ LogUniform(1.5/4, 4/1.5)``
  per instance and a jitter ``eta_i ~ LogUniform(1/1.5, 1.5)`` per noise group,
* the controller's estimate ``p_hat_i = max(0, N(p_i, (p_i/10)^2))`` where the controller uses one,
* the degeneracy screen (100 random controllers, keep the instance if at least 25 of them succeed
  in each of 10 repeated trials),
* the single-trajectory prior with its cap on unsuccessful trajectories, and
* the trajectory-set prior with exact crash and success counts and random cutoffs.

Every random draw comes from the generator passed in.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from esbopfn.core import Instance, ProblemSpec, Trajectory, Trial
from esbopfn.util.seeds import substream

if TYPE_CHECKING:
    from esbopfn.problems.base import Problem

log = logging.getLogger(__name__)

#: Bounds of the per-instance noise level ``S``.
NOISE_LEVEL = (1.5 / 4.0, 4.0 / 1.5)
#: Half-width factor of the per-group noise jitter ``eta``.
NOISE_JITTER = 1.5


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


@dataclass(frozen=True)
class ScaleParam:
    """A positive magnitude drawn log-uniformly on ``[nominal / f_lo, nominal * f_hi]``."""

    name: str
    nominal: float
    f_lo: float
    f_hi: float

    @property
    def lo(self) -> float:
        return self.nominal / self.f_lo

    @property
    def hi(self) -> float:
        return self.nominal * self.f_hi


@dataclass(frozen=True)
class BoundedParam:
    """A parameter drawn uniformly on its physical domain ``[lo, hi]``."""

    name: str
    lo: float
    hi: float
    nominal: float | None = None

    def __post_init__(self) -> None:
        if self.hi < self.lo:
            raise ValueError(f"{self.name}: hi must not be below lo")


@dataclass(frozen=True)
class TiedParam:
    """A parameter drawn relative to another one, ``value[source] * LogUniform(1/zeta, zeta)``."""

    name: str
    source: str
    zeta: float = 1.5
    nominal: float | None = None


Param = ScaleParam | BoundedParam | TiedParam


@dataclass(frozen=True)
class NoiseGroup:
    """Channels that share one noise base and one jitter draw.

    ``kind`` is ``sensor`` (added to the listed measured channels) or ``actuation`` (added to the
    applied input, the Furuta motor voltage). ``channels`` names what the noise perturbs.
    """

    name: str
    base: float
    channels: tuple[str, ...] = ()
    kind: str = "sensor"


@dataclass(frozen=True)
class PriorSpec:
    """The universal prior of one problem (thesis tables tab:app:prior-phi and tab:app:prior-noise).

    ``phi`` holds the entries of ``p`` in simulator order, ``extras`` per-instance quantities that
    are not part of ``p_hat`` (the Furuta dead time and reset kick).
    """

    problem: str
    phi: tuple[Param, ...]
    noise: tuple[NoiseGroup, ...]
    extras: tuple[BoundedParam, ...] = ()
    uses_phi_hat: bool = False
    estimate_sigma_rel: float = 0.10
    noise_level: tuple[float, float] = NOISE_LEVEL
    noise_jitter: float = NOISE_JITTER

    def __post_init__(self) -> None:
        names = [p.name for p in self.phi]
        for i, p in enumerate(self.phi):
            if isinstance(p, TiedParam) and p.source not in names[:i]:
                raise ValueError(f"{self.problem}: tied {p.name} needs an earlier source {p.source}")

    @property
    def phi_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.phi)

    @property
    def extra_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.extras)

    @property
    def noise_names(self) -> tuple[str, ...]:
        return tuple(g.name for g in self.noise)

    def param(self, name: str) -> Param:
        for p in (*self.phi, *self.extras):
            if p.name == name:
                return p
        raise KeyError(f"{self.problem}: no parameter {name!r}")

    def support(self, name: str) -> tuple[float, float]:
        """The support ``[lo, hi]`` of a parameter (for Tied: the source support times zeta)."""
        p = self.param(name)
        if isinstance(p, TiedParam):
            lo, hi = self.support(p.source)
            return lo / p.zeta, hi * p.zeta
        return float(p.lo), float(p.hi)

    def noise_support(self, name: str) -> tuple[float, float]:
        """The support of ``sigma`` for one noise group, ``[base / 4, 4 * base]``."""
        g = next(g for g in self.noise if g.name == name)
        lo = self.noise_level[0] / self.noise_jitter
        hi = self.noise_level[1] * self.noise_jitter
        return g.base * lo, g.base * hi

    def nominal_phi(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for p in self.phi:
            if isinstance(p, ScaleParam):
                out[p.name] = float(p.nominal)
            elif p.nominal is not None:
                out[p.name] = float(p.nominal)
            elif isinstance(p, TiedParam):
                out[p.name] = out[p.source]
            else:
                out[p.name] = 0.5 * (p.lo + p.hi)
        return out

    def nominal_instance(self, noise: bool = True, seed: int = 0) -> Instance:
        """The default plant: nominal ``p``, exact estimate, base noise (or none), nominal extras."""
        phi = self.nominal_phi()
        extras = {p.name: float(p.nominal if p.nominal is not None else 0.5 * (p.lo + p.hi))
                  for p in self.extras}
        noise_std = {g.name: (float(g.base) if noise else 0.0) for g in self.noise}
        return Instance(phi=phi, phi_hat=dict(phi) if self.uses_phi_hat else {},
                        noise_std=noise_std, extras=extras, seed=seed)

    def with_extra(self, name: str, value: float) -> PriorSpec:
        """A copy whose extra ``name`` is fixed at ``value`` (the Furuta no-dead-time ablation)."""
        if name not in self.extra_names:
            raise KeyError(f"{self.problem}: no extra {name!r}")
        extras = tuple(BoundedParam(p.name, value, value, value) if p.name == name else p
                       for p in self.extras)
        return replace(self, extras=extras)


def _prior_of(problem: Problem | PriorSpec) -> PriorSpec:
    return problem if isinstance(problem, PriorSpec) else problem.prior


def sample_instance(problem: Problem | PriorSpec, rng: np.random.Generator) -> Instance:
    """Draw one system instance ``(p, p_hat, sigma, extras)`` from the universal prior."""
    prior = _prior_of(problem)
    phi: dict[str, float] = {}
    for p in prior.phi:
        if isinstance(p, ScaleParam):
            phi[p.name] = _log_uniform(rng, p.lo, p.hi)
        elif isinstance(p, BoundedParam):
            phi[p.name] = float(rng.uniform(p.lo, p.hi))
        else:
            phi[p.name] = phi[p.source] * _log_uniform(rng, 1.0 / p.zeta, p.zeta)
    extras = {p.name: float(rng.uniform(p.lo, p.hi)) for p in prior.extras}
    level = _log_uniform(rng, *prior.noise_level)
    noise_std = {g.name: g.base * level * _log_uniform(rng, 1.0 / prior.noise_jitter,
                                                         prior.noise_jitter)
                 for g in prior.noise}
    phi_hat: dict[str, float] = {}
    if prior.uses_phi_hat:
        for name, value in phi.items():
            phi_hat[name] = max(0.0, float(rng.normal(value, prior.estimate_sigma_rel * value)))
    seed = int(rng.integers(0, 2**63 - 1, dtype=np.int64))
    return Instance(phi=phi, phi_hat=phi_hat, noise_std=noise_std, extras=extras, seed=seed)


def sample_theta(spec: ProblemSpec, rng: np.random.Generator) -> np.ndarray:
    """Draw ``theta ~ U(Theta)``."""
    return rng.uniform(spec.theta_lo, spec.theta_hi)


def screen(problem: Problem, instance: Instance, rng: np.random.Generator, n_theta: int = 100,
           repeats: int = 10, min_pass: int = 25, t_max: int | None = None) -> bool:
    """The degeneracy screen: keep ``instance`` iff at least ``min_pass`` of ``n_theta`` random
    controllers succeed in all ``repeats`` repeated trials.

    Stops as soon as the decision is fixed: after ``min_pass`` passing controllers, or when the
    remaining controllers can no longer reach ``min_pass``, and a controller is dropped at its
    first unsuccessful repeat.
    """
    passed = 0
    for i in range(n_theta):
        if passed >= min_pass:
            return True
        if passed + (n_theta - i) < min_pass:
            return False
        theta = sample_theta(problem.spec, rng)
        if all(problem.simulate(theta, instance, rng, t_max).success for _ in range(repeats)):
            passed += 1
    return passed >= min_pass


@dataclass(frozen=True)
class Pilot:
    """A seeded pilot run that estimates a rate from ``n`` prior draws."""

    n: int = 2000
    seed: int = 0


@dataclass
class PriorSample:
    """One draw of the single-trajectory prior."""

    instance: Instance
    trajectory: Trajectory


def unsuccessful_reject_probability(rho: float, max_share: float = 0.5) -> float:
    """Probability of rejecting an unsuccessful trajectory so that at most ``max_share`` of the
    accepted ones are unsuccessful, ``max(0, 1 - q (1 - rho) / ((1 - q) rho))``. For ``q = 1/2``
    this is the thesis formula ``max(0, 1 - (1 - rho) / rho)``."""
    if rho <= 0.0:
        return 0.0
    if rho >= 1.0:
        raise ValueError("every pilot trajectory was unsuccessful, the cap cannot be met")
    q = max_share
    return max(0.0, 1.0 - q * (1.0 - rho) / ((1.0 - q) * rho))


def estimate_unsuccessful_rate(problem: Problem, pilot: Pilot, t_max: int | None = None) -> float:
    """Share of unsuccessful trajectories under the unguarded single-trajectory prior."""
    rng = substream(pilot.seed, "pilot", problem.name)
    bad = 0
    for _ in range(pilot.n):
        inst = sample_instance(problem, rng)
        theta = sample_theta(problem.spec, rng)
        bad += not problem.simulate(theta, inst, rng, t_max).success
    rho = bad / pilot.n
    log.info("%s: pilot unsuccessful rate %.4f over %d draws", problem.name, rho, pilot.n)
    return rho


@dataclass
class SingleTrajectorySampler:
    """The single-trajectory prior of one problem, with its cached unsuccessful rate ``rho``.

    Each draw samples an instance and ``theta ~ U(Theta)``, simulates, and rejects an unsuccessful
    trajectory with probability :attr:`reject_prob`. No degeneracy screen is applied.
    """

    problem: Problem
    rng: np.random.Generator
    rho: float
    max_unsuccessful_share: float = 0.5
    t_max: int | None = None
    n_simulated: int = field(default=0, init=False)
    n_accepted: int = field(default=0, init=False)

    @property
    def reject_prob(self) -> float:
        return unsuccessful_reject_probability(self.rho, self.max_unsuccessful_share)

    def sample(self) -> PriorSample:
        reject = self.reject_prob
        while True:
            inst = sample_instance(self.problem, self.rng)
            theta = sample_theta(self.problem.spec, self.rng)
            traj = self.problem.simulate(theta, inst, self.rng, self.t_max)
            self.n_simulated += 1
            if traj.success or self.rng.random() >= reject:
                self.n_accepted += 1
                return PriorSample(inst, traj)

    def __iter__(self) -> Iterator[PriorSample]:
        while True:
            yield self.sample()


def single_trajectory_prior(problem: Problem, rng: np.random.Generator,
                            max_unsuccessful_share: float = 0.5, pilot: Pilot = Pilot(),
                            rho: float | None = None,
                            t_max: int | None = None) -> SingleTrajectorySampler:
    """Return the single-trajectory prior sampler of ``problem``.

    ``rho`` (the unsuccessful rate) is estimated once by the seeded ``pilot`` unless given. Problems
    that can neither crash nor fail a success criterion skip the pilot (``rho = 0``).
    """
    if rho is None:
        if not (problem.spec.can_crash or problem.spec.has_success):
            rho = 0.0
        else:
            rho = estimate_unsuccessful_rate(problem, pilot, t_max)
    return SingleTrajectorySampler(problem, rng, float(rho), max_unsuccessful_share, t_max)


@dataclass(frozen=True)
class SetRates:
    """Per-instance crash rate and unsuccessful-given-not-crashed rate."""

    rho_c: float
    rho_sbar_given_cbar: float


def estimate_set_rates(problem: Problem, instance: Instance, rng: np.random.Generator,
                       n_pilot: int = 100, t_max: int | None = None) -> SetRates:
    """Estimate ``rho_c`` and ``rho_{sbar|cbar}`` on ``instance`` from ``n_pilot`` random controllers."""
    crashed = unsuccessful = 0
    for _ in range(n_pilot):
        traj = problem.simulate(sample_theta(problem.spec, rng), instance, rng, t_max)
        if traj.crashed:
            crashed += 1
        elif not traj.success:
            unsuccessful += 1
    rho_c = crashed / n_pilot
    not_crashed = n_pilot - crashed
    return SetRates(rho_c, unsuccessful / not_crashed if not_crashed else 0.0)


@dataclass
class TrajectorySet:
    """One draw of the trajectory-set prior: the trials and the counts that shaped them."""

    trials: list[Trial]
    n_crashed: int
    n_unsuccessful: int
    n_successful: int
    p_cut: float
    n_simulated: int


def trajectory_set_prior(problem: Problem, instance: Instance, n: int, rng: np.random.Generator,
                         rates: SetRates | None = None, n_pilot: int = 100, cut: bool = True,
                         max_draws: int | None = None, t_max: int | None = None) -> TrajectorySet:
    """Draw one example of the trajectory-set prior on a (screened) ``instance``.

    Draws ``rho'_c ~ U(0, rho_c)`` and ``rho'_{sbar|cbar} ~ U(0, rho_{sbar|cbar})``, the counts
    ``n_c ~ Bin(n, rho'_c)`` and ``n_{sbar,cbar} ~ Bin(n - n_c, rho'_{sbar|cbar})``, and rejects random
    controllers until every count is met exactly. With ``cut``, one probability ``~U(0, 1)`` is drawn
    and each trial is independently cut with it at ``~U{1..T_i}`` (``T_i`` its own length). The
    rates come from a pilot of ``n_pilot`` controllers unless given.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if rates is None:
        rates = estimate_set_rates(problem, instance, rng, n_pilot, t_max)
    rho_c = rng.uniform(0.0, rates.rho_c)
    rho_s = rng.uniform(0.0, rates.rho_sbar_given_cbar)
    n_c = int(rng.binomial(n, rho_c))
    n_sbar = int(rng.binomial(n - n_c, rho_s))
    need = {"crashed": n_c, "unsuccessful": n_sbar, "successful": n - n_c - n_sbar}
    limit = max_draws if max_draws is not None else 1000 * n
    trajs: list[Trajectory] = []
    draws = 0
    while len(trajs) < n:
        if draws >= limit:
            raise RuntimeError(f"{problem.name}: trajectory-set rejection exceeded {limit} draws "
                               f"(still needed {need})")
        traj = problem.simulate(sample_theta(problem.spec, rng), instance, rng, t_max)
        draws += 1
        kind = "crashed" if traj.crashed else ("successful" if traj.success else "unsuccessful")
        if need[kind] > 0:
            need[kind] -= 1
            trajs.append(traj)
    p_cut = float(rng.uniform()) if cut else 0.0
    trials = []
    for traj in trajs:
        length = traj.crash_step if traj.crash_step is not None else traj.T
        t_obs = int(rng.integers(1, length + 1)) if cut and rng.random() < p_cut else traj.T
        trials.append(traj.cut(t_obs))
    return TrajectorySet(trials, n_c, n_sbar, n - n_c - n_sbar, p_cut, draws)
