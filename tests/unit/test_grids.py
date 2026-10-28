import math

import numpy as np
import pytest

from esbopfn.core import Trajectory
from esbopfn.grids import N_REF, curve_grid, decoder_grid, n_grid, stride, tokens
from esbopfn.problems.registry import load_problem_specs

THESIS_TABLE = {  # tab:app:grid
    "boiler": (701, 11, 64),
    "threetank_pi": (476, 7, 68),
    "pt2_deadtime": (501, 8, 63),
    "cartpole": (1000, 15, 67),
    "furuta": (600, 9, 67),
}


@pytest.mark.parametrize("name", list(THESIS_TABLE))
def test_thesis_grid_table(name):
    spec = load_problem_specs()[name]
    T, s, n = THESIS_TABLE[name]
    assert spec.T == T and stride(spec) == s and n_grid(spec) == n
    g = decoder_grid(spec)
    assert len(g) == n - 1 and g[0] == s and g[-1] < T
    np.testing.assert_array_equal(np.diff(g), s)
    c = curve_grid(spec, n)
    assert len(c) == n and c[-1] == T and np.all(np.diff(c) > 0)


def test_reference_and_ties():
    assert N_REF == 67
    # T = 134: s = 2 gives exactly 67
    assert stride(134) == 2
    # T = 89: s = 1 gives 89 and s = 2 gives 45, both 22 away from 67, the tie goes to 1
    assert stride(89) == 1
    # T = 2000: s = 30 gives exactly 67
    assert stride(2000) == 30


def test_curve_grid_formula():
    np.testing.assert_array_equal(curve_grid(10, 3), [3, 6, 10])
    np.testing.assert_array_equal(curve_grid(1000, 8), [125, 250, 375, 500, 625, 750, 875, 1000])


def _synthetic(T=12, crash_step=None, with_rates=True):
    t = np.arange(1, T + 1, dtype=np.float64)
    j = t.copy()                              # stage cost t, so J_t = t(t+1)/2
    J = np.cumsum(j)
    if crash_step is not None:
        J[crash_step - 1:] = np.inf
    x = np.stack([10 * t, -t], axis=1)        # two level channels
    xdot = np.stack([np.ones(T), 2 * t], axis=1) if with_rates else None
    return Trajectory(theta=np.zeros(1), J=J, x=x, xdot=xdot, crash_step=crash_step,
                      success=crash_step is None)


def test_tokens_known_sums():
    traj = _synthetic()
    tok = tokens(traj, np.array([3, 6, 9]), t_obs=12)
    np.testing.assert_array_equal(tok.t, [3, 6, 9])
    np.testing.assert_allclose(tok.J, [6, 21, 45])
    cov = tok.covariates
    assert cov.shape == (3, 1 + 2 + 2)
    np.testing.assert_allclose(cov[:, 0], [6, 15, 24])                  # J differences
    np.testing.assert_allclose(cov[:, 1:3], [[30, -3], [60, -6], [90, -9]])  # levels at t_l
    np.testing.assert_allclose(cov[:, 3], [3, 3, 3])                    # sum of ones
    np.testing.assert_allclose(cov[:, 4], [2 * 6, 2 * 15, 2 * 24])      # sum of 2t per interval


def test_tokens_cut_and_crash():
    traj = _synthetic()
    assert len(tokens(traj, np.array([3, 6, 9]), t_obs=7)) == 2
    assert len(tokens(traj, np.array([3, 6, 9]), t_obs=2)) == 0
    crashed = _synthetic(crash_step=6)
    tok = tokens(crashed, np.array([3, 6, 9]), t_obs=12)
    np.testing.assert_array_equal(tok.t, [3])
    assert np.all(np.isfinite(tok.covariates))


def test_tokens_difference_rates():
    traj = _synthetic(with_rates=False)
    cov = tokens(traj, np.array([3, 6]), t_obs=12).covariates
    # level differences summed: x_{t_l} - x_{t_{l-1}}, with a zero first step
    np.testing.assert_allclose(cov[:, 3:5], [[20, -2], [30, -3]])
    assert math.isclose(cov[0, 0], 6)
