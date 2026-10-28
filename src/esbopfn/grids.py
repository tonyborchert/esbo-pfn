"""Subsampling grids and DecPFN token covariates (thesis Appendix, subsampling grids).

All steps are 1-based. The decoder grid uses an exact stride derived from the Cartpole reference
(stride 15 on ``T = 1000``, so ``N_ref = 67``): every problem takes the stride whose
``N_grid = ceil(T / stride)`` is closest to ``N_ref``, ties going to the smaller stride.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from esbopfn.core import ProblemSpec, Trajectory

REFERENCE_T = 1000
REFERENCE_STRIDE = 15
N_REF = math.ceil(REFERENCE_T / REFERENCE_STRIDE)

#: Number of points of the FT-PFN curve grid.
FTPFN_N_GRID = 8


def _horizon(spec_or_T: ProblemSpec | int) -> int:
    T = spec_or_T.T if isinstance(spec_or_T, ProblemSpec) else int(spec_or_T)
    if T < 2:
        raise ValueError(f"horizon must be at least 2, got {T}")
    return T


def stride(spec_or_T: ProblemSpec | int) -> int:
    """Decoder stride ``argmin_s |ceil(T / s) - N_ref|``, ties to the smaller ``s``."""
    T = _horizon(spec_or_T)
    return min(range(1, T + 1), key=lambda s: (abs(math.ceil(T / s) - N_REF), s))


def n_grid(spec_or_T: ProblemSpec | int) -> int:
    """Number of grid points ``ceil(T / stride)``, the horizon included."""
    T = _horizon(spec_or_T)
    return math.ceil(T / stride(T))


def decoder_grid(spec_or_T: ProblemSpec | int) -> np.ndarray:
    """Token steps of DecPFN and XDecPFN, ``t_l = l * stride`` for ``l = 1 .. N_grid - 1``.

    The horizon ``T`` is the prediction target and is not part of the returned array.
    """
    T = _horizon(spec_or_T)
    s = stride(T)
    return np.arange(1, n_grid(T), dtype=np.int64) * s


def curve_grid(spec_or_T: ProblemSpec | int, N: int) -> np.ndarray:
    """Approximately equidistant grid ``t_l = floor(l * T / N)`` for ``l = 1 .. N``.

    Used by LC-PFN (``N = N_grid``), Trajectory-TabPFN (``N = L``) and FT-PFN (``N = 8``).
    The last point is always ``T``.
    """
    T = _horizon(spec_or_T)
    if not 1 <= N <= T:
        raise ValueError(f"N must lie in [1, {T}], got {N}")
    ell = np.arange(1, N + 1, dtype=np.int64)
    return (ell * T) // N


@dataclass
class Tokens:
    """Per-grid-point quantities of one observed trajectory.

    ``covariates[l] = [J_{t_l} - J_{t_{l-1}}, xi_{t_l}, sum of xi_dot over (t_{l-1}, t_l]]`` with
    ``t_0 = 0`` and ``J_0 = 0``. Its width is ``1 + C + C_rate``, where ``C_rate`` is the number of
    measured rate channels or ``C`` when the rates are differences of the levels.
    """

    t: np.ndarray
    J: np.ndarray
    covariates: np.ndarray

    def __len__(self) -> int:
        return int(self.t.shape[0])


def _rates(traj: Trajectory) -> np.ndarray:
    """Per-step rates: measured if present, else ``xi_t - xi_{t-1}`` with a zero first step."""
    if traj.xdot is not None:
        return np.asarray(traj.xdot, dtype=np.float64)
    x = np.asarray(traj.x, dtype=np.float64)
    diff = np.zeros_like(x)
    diff[1:] = x[1:] - x[:-1]
    return diff


def tokens(traj: Trajectory, grid: np.ndarray, t_obs: int) -> Tokens:
    """Covariates for every grid point ``t_l <= t_obs`` that lies before a crash.

    Rates are summed over each interval at the raw step resolution, levels are read at the grid
    point. A grid point at or after the crash step is not tokenized (its cost is infinite).
    """
    grid = np.asarray(grid, dtype=np.int64)
    if grid.ndim != 1 or np.any(np.diff(grid) <= 0) or (grid.size and grid[0] < 1):
        raise ValueError("grid must be a strictly increasing array of 1-based steps")
    limit = min(int(t_obs), traj.T)
    if traj.crash_step is not None:
        limit = min(limit, traj.crash_step - 1)
    t = grid[grid <= limit]
    J = np.asarray(traj.J, dtype=np.float64)
    x = np.asarray(traj.x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    rates = _rates(traj)
    if rates.ndim == 1:
        rates = rates[:, None]
    width = 1 + x.shape[1] + rates.shape[1]
    if t.size == 0:
        return Tokens(t=t, J=np.zeros(0), covariates=np.zeros((0, width)))
    J_t = J[t - 1]
    dJ = np.diff(np.concatenate([[0.0], J_t]))
    csum = np.concatenate([np.zeros((1, rates.shape[1])), np.cumsum(rates, axis=0)])
    prev = np.concatenate([[0], t[:-1]])
    rate_sums = csum[t] - csum[prev]
    cov = np.concatenate([dJ[:, None], x[t - 1], rate_sums], axis=1)
    return Tokens(t=t, J=J_t, covariates=cov)
