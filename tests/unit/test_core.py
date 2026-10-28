import math

import numpy as np
import pytest

from esbopfn.core import ProblemSpec, Trajectory
from esbopfn.problems.registry import PROBLEM_NAMES, get_spec, load_problem_specs


def make_traj(T=10, crash_step=None, success=True):
    J = np.cumsum(np.ones(T, dtype=np.float32))
    if crash_step is not None:
        J[crash_step - 1:] = np.inf
    return Trajectory(theta=np.zeros(2), J=J, x=np.zeros((T, 1)), xdot=None,
                      crash_step=crash_step, success=success and crash_step is None)


def test_complete_trial():
    tr = make_traj().cut(10)
    assert tr.complete and not tr.crashed and not tr.early_stopped and tr.successful
    assert tr.J_obs == 10.0


def test_early_stopped_trial():
    tr = make_traj().cut(4)
    assert tr.early_stopped and not tr.complete and not tr.crashed
    assert tr.J_obs == 4.0


def test_crashed_trial_and_clamp():
    traj = make_traj(crash_step=6)
    tr = traj.cut(10)
    assert tr.t_obs == 6 and tr.crashed and not tr.complete and not tr.early_stopped
    assert math.isinf(tr.J_obs)
    before = traj.cut(5)
    assert not before.crashed and before.early_stopped and before.J_obs == 5.0


def test_unrun_trial():
    tr = make_traj().cut(0)
    assert tr.unrun and tr.J_obs == 0.0 and not tr.early_stopped


def test_cut_bounds():
    with pytest.raises(ValueError):
        make_traj().cut(11)


def test_problem_specs():
    specs = load_problem_specs()
    assert tuple(specs) == PROBLEM_NAMES
    expected = {  # name: (d, T, can_crash, has_success, n_max, t_budget, n_init)
        "boiler": (1, 701, False, False, 45, 10515, 2),
        "threetank_pi": (2, 476, False, False, 90, 14280, 2),
        "pt2_deadtime": (3, 501, True, False, 135, 22545, 3),
        "cartpole": (4, 1000, True, False, 180, 60000, 4),
        "furuta": (8, 600, True, True, 180, 36000, 8),
    }
    for name, (d, T, crash, succ, n_max, t_budget, n_init) in expected.items():
        s = specs[name]
        assert (s.d, s.T, s.can_crash, s.has_success) == (d, T, crash, succ)
        assert (s.n_max(), s.t_budget(), s.n_init()) == (n_max, t_budget, n_init)
        assert len(s.theta_names) == d
    cp = specs["cartpole"]
    np.testing.assert_allclose(cp.theta_lo, [0.3, 30, 10, 150])
    np.testing.assert_allclose(cp.theta_hi, [3.0, 300, 100, 1500])
    assert cp.rate_channels == ("xdot_cart", "phidot_pole")


def test_spec_equality_and_hash():
    a, b = get_spec("boiler"), get_spec("boiler")
    assert a == b and hash(a) == hash(b) and a != get_spec("cartpole")
    with pytest.raises(ValueError):
        ProblemSpec("x", 1, np.array([1.0]), np.array([0.0]), 10, ("a",), (), False, False)
    with pytest.raises(KeyError):
        get_spec("nope")
