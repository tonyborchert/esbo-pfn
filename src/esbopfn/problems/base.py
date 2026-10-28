"""The :class:`Problem` bundle and helpers shared by the five simulators.

A simulator is a function ``simulate(theta, instance, rng, t_max=None) -> Trajectory``. It runs the
closed loop for ``t_max`` logged steps (the full horizon ``spec.T`` by default) and draws every
random number (start state, sensor and actuation noise, reset kick) from ``rng``.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from esbopfn.core import Instance, ProblemSpec, Trajectory

if TYPE_CHECKING:
    from esbopfn.problems.prior import PriorSpec


class Simulator(Protocol):
    def __call__(self, theta: np.ndarray, instance: Instance, rng: np.random.Generator,
                 t_max: int | None = None) -> Trajectory: ...


@dataclass(frozen=True)
class Problem:
    """One control problem: its static spec, its universal prior and its simulator."""

    spec: ProblemSpec
    prior: PriorSpec
    simulator: Callable[..., Trajectory]

    @property
    def name(self) -> str:
        return self.spec.name

    def simulate(self, theta: np.ndarray, instance: Instance, rng: np.random.Generator,
                 t_max: int | None = None) -> Trajectory:
        """Run one trial of controller ``theta`` on ``instance``."""
        return self.simulator(theta, instance, rng, t_max)

    def nominal_instance(self, noise: bool = True) -> Instance:
        """The default plant with an exact estimate and the base noise levels (or no noise)."""
        return self.prior.nominal_instance(noise=noise)


def horizon(spec: ProblemSpec, t_max: int | None) -> int:
    """Number of logged steps to simulate, validated against the spec."""
    if t_max is None:
        return spec.T
    t_max = int(t_max)
    if not 1 <= t_max <= spec.T:
        raise ValueError(f"{spec.name}: t_max must lie in [1, {spec.T}], got {t_max}")
    return t_max


def check_theta(spec: ProblemSpec, theta: np.ndarray) -> np.ndarray:
    """Return ``theta`` as a float64 vector of length ``spec.d``."""
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    if theta.shape != (spec.d,):
        raise ValueError(f"{spec.name}: theta must have {spec.d} entries, got {theta.shape[0]}")
    return theta


def finish(theta: np.ndarray, j: np.ndarray, x: np.ndarray, xdot: np.ndarray | None, T: int,
           crash_step: int | None, success: bool | None = None) -> Trajectory:
    """Assemble a :class:`Trajectory` from per-step costs of the simulated steps.

    ``j``, ``x`` and ``xdot`` may be shorter than ``T`` when the trial crashed. The cumulative cost
    is ``+inf`` from ``crash_step`` on, and the channels past the last simulated step repeat its
    value so the arrays keep their full length. ``success`` defaults to ``not crashed``.
    """
    n = len(j)
    J = np.full(T, math.inf, dtype=np.float64)
    J[:n] = np.cumsum(np.asarray(j, dtype=np.float64))
    if crash_step is not None:
        J[crash_step - 1:] = math.inf
    xs = _pad(x, T)
    xds = None if xdot is None else _pad(xdot, T)
    crashed = crash_step is not None
    return Trajectory(theta=np.asarray(theta, dtype=np.float64), J=J.astype(np.float32), x=xs,
                      xdot=xds, crash_step=crash_step,
                      success=(not crashed) if success is None else bool(success and not crashed))


def _pad(a: np.ndarray, T: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]
    if a.shape[0] == T:
        return a
    out = np.empty((T, a.shape[1]), dtype=np.float32)
    out[: a.shape[0]] = a
    out[a.shape[0]:] = a[-1] if a.shape[0] else 0.0
    return out


@functools.cache
def cached_spec(name: str) -> ProblemSpec:
    """The spec of ``name`` from the packaged ``problems.yaml``, loaded on first use."""
    from esbopfn.problems.registry import get_spec

    return get_spec(name)


@functools.cache
def cached_prior(name: str) -> PriorSpec:
    """The universal prior of ``name`` from the packaged ``problems.yaml``, loaded on first use."""
    from esbopfn.problems.registry import get_prior

    return get_prior(name)
