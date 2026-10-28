"""Simulator contracts: shapes, determinism, crash rules, cost formulas, success rule."""

from __future__ import annotations

import math

import numpy as np
import pytest

from esbopfn.problems import PROBLEM_NAMES, get_problem
from esbopfn.problems.furuta import sim as furuta_sim
from esbopfn.problems.prior import sample_instance, sample_theta
from esbopfn.problems.pt2_deadtime import sim as pt2_sim


def _noiseless(name: str):
    problem = get_problem(name)
    return problem, problem.nominal_instance(noise=False)


@pytest.mark.parametrize("name", PROBLEM_NAMES)
def test_shapes_and_spec(name: str) -> None:
    problem = get_problem(name)
    spec = problem.spec
    rng = np.random.default_rng(0)
    traj = problem.simulate(sample_theta(spec, rng), sample_instance(problem, rng), rng)
    assert traj.J.shape == (spec.T,) and traj.J.dtype == np.float32
    assert traj.x.shape == (spec.T, len(spec.channels)) and traj.x.dtype == np.float32
    if spec.rate_channels:
        assert traj.xdot.shape == (spec.T, len(spec.rate_channels))
    else:
        assert traj.xdot is None
    assert np.isfinite(traj.x).all()
    finite = traj.T if traj.crash_step is None else traj.crash_step - 1
    assert np.isfinite(traj.J[:finite]).all() and (np.diff(traj.J[:finite]) >= 0).all()
    if not spec.has_success:
        assert traj.success == (not traj.crashed)
    for mod in ("SPEC", "PRIOR", "simulate"):
        assert hasattr(__import__(f"esbopfn.problems.{name}", fromlist=[mod]), mod)


@pytest.mark.parametrize("name", PROBLEM_NAMES)
def test_deterministic_given_seed(name: str) -> None:
    problem = get_problem(name)
    inst = sample_instance(problem, np.random.default_rng(1))
    theta = sample_theta(problem.spec, np.random.default_rng(2))
    a = problem.simulate(theta, inst, np.random.default_rng(3))
    b = problem.simulate(theta, inst, np.random.default_rng(3))
    c = problem.simulate(theta, inst, np.random.default_rng(4))
    np.testing.assert_array_equal(a.J, b.J)
    np.testing.assert_array_equal(a.x, b.x)
    assert a.crash_step == b.crash_step and a.success == b.success
    assert not np.array_equal(a.x, c.x)  # the noise differs between seeds


@pytest.mark.parametrize("name", PROBLEM_NAMES)
def test_t_max_truncates(name: str) -> None:
    problem, inst = _noiseless(name)
    theta = 0.5 * (problem.spec.theta_lo + problem.spec.theta_hi)
    full = problem.simulate(theta, inst, np.random.default_rng(0))
    short = problem.simulate(theta, inst, np.random.default_rng(0), t_max=50)
    assert short.T == 50
    if full.crash_step is None or full.crash_step > 50:
        np.testing.assert_allclose(short.J, full.J[:50], rtol=1e-6)
    with pytest.raises(ValueError):
        problem.simulate(theta, inst, np.random.default_rng(0), t_max=problem.spec.T + 1)


def test_boiler_cost() -> None:
    problem = get_problem("boiler")
    traj = problem.simulate(np.array([60.0]), problem.nominal_instance(), np.random.default_rng(0))
    meas = traj.x[:, 0].astype(np.float64)
    j = np.diff(np.concatenate([[0.0], traj.J.astype(np.float64)]))
    comfort = 0.5 * np.minimum(np.abs(20.0 - meas), 2.0) ** 2
    rest = j - comfort
    # energy (u in {0, 1}) plus the switching penalty 8 when the burner turns on, 0 at t = 1
    assert np.allclose(rest[np.abs(rest) < 0.5], 0.0, atol=1e-3)
    assert np.allclose(rest[(rest > 0.5) & (rest < 1.5)], 1.0, atol=1e-3)
    assert np.allclose(rest[rest > 1.5], 9.0, atol=1e-3)
    assert rest[0] < 1.5
    # ADC quantization: every reading lies on the 0.390625 degC grid offset by -15 degC
    q = (meas + 15.0) / 0.390625
    np.testing.assert_allclose(q, np.round(q), atol=1e-6)
    assert not traj.crashed and traj.success


def test_threetank_cost() -> None:
    problem = get_problem("threetank_pi")
    traj = problem.simulate(np.array([2.0, 0.05]), problem.nominal_instance(),
                            np.random.default_rng(0))
    j = (3e-3 - traj.x[:, 2].astype(np.float64)) ** 2
    np.testing.assert_allclose(traj.J, np.cumsum(j), rtol=1e-5)
    # the logged B104 level is the noisy measurement the cost uses
    clean = problem.simulate(np.array([2.0, 0.05]), problem.nominal_instance(noise=False),
                             np.random.default_rng(0))
    assert np.abs(traj.x[:5, 2] - clean.x[:5, 2]).max() > 1e-5


def test_pt2_cost_and_crash_rule() -> None:
    problem, inst = _noiseless("pt2_deadtime")
    good = problem.simulate(np.log10([5.0, 2.0, 1.0]), inst, np.random.default_rng(0))
    y = good.x[:, 0].astype(np.float64)
    tau = np.arange(problem.spec.T) * 0.1
    np.testing.assert_allclose(good.J, np.cumsum(tau * np.abs(1.0 - y)), rtol=1e-5)
    assert not good.crashed and (y - 1.0 <= 2.0).all()

    bad = problem.simulate(np.array([0.87, 0.63, -0.99]), inst, np.random.default_rng(0))
    k = bad.crash_step
    assert k is not None and not bad.success
    yb = bad.x[:, 0]
    assert yb[k - 1] - 1.0 > 2.0 and (yb[: k - 1] - 1.0 <= 2.0).all()
    assert np.isinf(bad.J[k - 1:]).all() and np.isfinite(bad.J[: k - 1]).all()


def test_pt2_dead_time_is_tenth_of_a_second() -> None:
    assert pt2_sim.DEAD_TIME_STEPS * pt2_sim.DT == pytest.approx(0.1)


def test_cartpole_cost_and_crash_rule() -> None:
    problem, inst = _noiseless("cartpole")
    rng = np.random.default_rng(0)
    goals = np.array([0.0] * 100 + [2.0] * 200 + [-2.0] * 300 + [2.0] * 200 + [-2.0] * 200)
    crashed = 0
    for _ in range(8):
        traj = problem.simulate(sample_theta(problem.spec, rng), inst, rng)
        n = traj.crash_step or traj.T
        x, phi = traj.x[:n, 0].astype(np.float64), traj.x[:n, 1].astype(np.float64)
        xd, phid = traj.xdot[:n, 0].astype(np.float64), traj.xdot[:n, 1].astype(np.float64)
        j = 10 * np.tanh(goals[:n] - x) ** 2 + xd**2 + phi**2 + phid**2
        finite = n if traj.crash_step is None else n - 1
        np.testing.assert_allclose(traj.J[:finite], np.cumsum(j)[:finite], rtol=1e-4)
        out = (np.abs(x) > 3.5) | (np.abs(phi) > math.radians(30))
        if traj.crash_step is None:
            assert not out.any()
        else:
            crashed += 1
            assert out[-1] and not out[:-1].any()
    assert crashed > 0


def test_furuta_cost_and_success_rule() -> None:
    problem = get_problem("furuta")
    rng = np.random.default_rng(1)
    inst = sample_instance(problem, np.random.default_rng(18))  # a screened instance
    up = math.radians(5.0)
    seen = set()
    for _ in range(12):
        traj = problem.simulate(sample_theta(problem.spec, rng), inst, rng)
        n = traj.crash_step or traj.T
        a = np.abs(traj.x[:n, 1].astype(np.float64))
        j = a * (1 + 20 * (1 - np.minimum(a, up) / up))
        finite = n if traj.crash_step is None else n - 1
        np.testing.assert_allclose(traj.J[:finite], np.cumsum(j)[:finite], rtol=1e-4)
        if traj.crash_step is None:
            assert traj.success == (a[-240:].mean() < up)
            seen.add(traj.success)
        else:
            assert not traj.success
            seen.add("crash")
    assert seen >= {True, "crash"}


def test_furuta_success_on_shortened_horizon() -> None:
    problem = get_problem("furuta")
    rng = np.random.default_rng(2)
    inst = problem.nominal_instance()
    traj = problem.simulate(sample_theta(problem.spec, rng), inst, rng, t_max=60)
    assert traj.T == 60
    if traj.crash_step is None:
        assert traj.success == (np.abs(traj.x[:, 1].astype(np.float64)).mean() < math.radians(5))


def test_furuta_undesignable_lqr_crashes_at_first_step(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("no stabilizing solution")

    monkeypatch.setattr(furuta_sim, "lqr_gains", fail)
    problem = get_problem("furuta")
    traj = problem.simulate(sample_theta(problem.spec, np.random.default_rng(0)),
                            problem.nominal_instance(), np.random.default_rng(0))
    assert traj.crash_step == 1 and not traj.success and np.isinf(traj.J).all()


def test_furuta_dead_time_line() -> None:
    line = furuta_sim.DeadTime(1.5)
    out = [line(v) for v in (1.0, 2.0, 3.0, 4.0)]
    assert out == pytest.approx([0.0, 0.5, 1.5, 2.5])
    ident = furuta_sim.DeadTime(0.0)
    assert [ident(v) for v in (1.0, -2.0)] == [1.0, -2.0]


def test_furuta_arm_crash_and_kick_sign() -> None:
    """A strong pump with no barrier winds the arm out of +-90 deg, and the reset kick moves the
    pendulum with a sign drawn per trial."""
    problem, inst = _noiseless("furuta")
    inst.extras["kick"] = 0.6
    starts = set()
    for seed in range(6):
        rng = np.random.default_rng(seed)
        traj = problem.simulate(np.array([90.0, 1.2, 12, 5, 1, 1, 1, 1.5]), inst, rng)
        starts.add(bool(traj.xdot[0, 1] > 0))
    assert starts == {True, False}
    crashes = [problem.simulate(np.array([90.0, 1.2, 12, 5, 1, 1, 1, 1.5]), inst,
                                np.random.default_rng(s)).crash_step for s in range(6)]
    assert any(c is not None for c in crashes)
