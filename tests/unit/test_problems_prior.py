"""The universal prior: supports (thesis tab:app:prior-phi, tab:app:prior-noise), estimate, screen,
single-trajectory prior and trajectory-set prior."""

from __future__ import annotations

import math

import numpy as np
import pytest

from esbopfn.problems import PROBLEM_NAMES, get_problem
from esbopfn.problems.prior import (
    Pilot,
    SetRates,
    sample_instance,
    sample_theta,
    screen,
    single_trajectory_prior,
    trajectory_set_prior,
    unsuccessful_reject_probability,
)
from esbopfn.problems.registry import load_priors

#: Supports as printed in tab:app:prior-phi (rounded to two or three digits there, hence the 2 % tolerance).
THESIS_SUPPORTS = {
    "boiler": {"heating_rate": (0.2, 5.0), "cooling_rate": (0.02, 0.5)},
    "threetank_pi": {"V101_opening": (10, 85), "V106_opening": (10, 90),
                     "B101_volume": (0.040, 0.130)},
    "pt2_deadtime": {"K": (0.8, 20.0), "w0": (0.1, 2.5), "D": (0.1, 2.5)},
    "cartpole": {"cart_mass": (0.125, 8.0), "pole_mass": (0.0125, 0.8),
                 "pole_half_length": (0.167, 1.5), "max_force": (3.33, 30.0)},
    "furuta": {"Rm": (2.1, 42.0), "kt": (0.0105, 0.21), "km": (0.007, 0.315),
               "mr": (0.024, 0.76), "Lr": (0.028, 0.34), "Dr": (0.275e-4, 8.25e-4),
               "mp": (0.006, 0.24), "Lp": (0.043, 0.516), "Dp": (0.0505e-4, 1.515e-4),
               "dead_time": (8.33e-3, 25.0e-3), "kick": (0.0, 0.6)},
}
#: Noise base levels of tab:app:prior-noise.
THESIS_NOISE = {
    "boiler": {"temperature": 0.03}, "threetank_pi": {"V4": 1.7e-4}, "pt2_deadtime": {"y": 4e-5},
    "cartpole": {"x_cart": 0.004, "xdot_cart": 0.02, "phi_pole": 0.002, "phidot_pole": 0.02},
    "furuta": {"angle": 0.002, "velocity": 0.07, "voltage": 0.1},
}


def test_priors_cover_every_problem() -> None:
    assert tuple(load_priors()) == PROBLEM_NAMES


@pytest.mark.parametrize("name", PROBLEM_NAMES)
def test_yaml_matches_thesis_tables(name: str) -> None:
    prior = get_problem(name).prior
    names = prior.phi_names + prior.extra_names
    assert set(names) == set(THESIS_SUPPORTS[name])
    for pname, (lo, hi) in THESIS_SUPPORTS[name].items():
        got = prior.support(pname)
        assert got == pytest.approx((lo, hi), rel=2e-2, abs=1e-12), pname
    assert {g.name: g.base for g in prior.noise} == THESIS_NOISE[name]
    for g in prior.noise:
        assert prior.noise_support(g.name) == pytest.approx((g.base / 4, 4 * g.base))
    assert prior.uses_phi_hat == (name == "furuta")


@pytest.mark.parametrize("name", PROBLEM_NAMES)
def test_sampled_instances_stay_in_support(name: str) -> None:
    problem = get_problem(name)
    prior = problem.prior
    rng = np.random.default_rng(0)
    insts = [sample_instance(problem, rng) for _ in range(2000)]
    for pname in prior.phi_names:
        lo, hi = prior.support(pname)
        vals = np.array([inst.phi[pname] for inst in insts])
        assert vals.min() >= lo and vals.max() <= hi, pname
        # log-uniform or uniform draws spread over most of the support
        assert vals.max() - vals.min() > 0.5 * (hi - lo) or vals.max() / vals.min() > 0.5 * hi / lo
    for pname in prior.extra_names:
        lo, hi = prior.support(pname)
        vals = np.array([inst.extras[pname] for inst in insts])
        assert vals.min() >= lo and vals.max() <= hi, pname
    for g in prior.noise:
        lo, hi = prior.noise_support(g.name)
        vals = np.array([inst.noise_std[g.name] for inst in insts])
        assert vals.min() >= lo and vals.max() <= hi
        assert vals.min() < g.base < vals.max()
    for inst in insts:
        if prior.uses_phi_hat:
            assert set(inst.phi_hat) == set(prior.phi_names)
            assert all(v >= 0 for v in inst.phi_hat.values())
        else:
            assert inst.phi_hat == {}
    assert len({inst.seed for inst in insts}) == len(insts)


def test_noise_level_shared_and_jitter_bounded() -> None:
    problem = get_problem("cartpole")
    rng = np.random.default_rng(1)
    for _ in range(500):
        inst = sample_instance(problem, rng)
        rel = [inst.noise_std[g.name] / g.base for g in problem.prior.noise]
        # every channel is S * eta with eta in [1/1.5, 1.5], so channels differ by at most 1.5^2
        assert max(rel) / min(rel) <= 1.5**2 + 1e-12


def test_furuta_tie_and_estimate() -> None:
    problem = get_problem("furuta")
    rng = np.random.default_rng(2)
    rel_err = []
    for _ in range(2000):
        inst = sample_instance(problem, rng)
        ratio = inst.phi["km"] / inst.phi["kt"]
        assert 1 / 1.5 <= ratio <= 1.5
        rel_err += [(inst.phi_hat[n] - inst.phi[n]) / inst.phi[n] for n in inst.phi]
    rel_err = np.array(rel_err)
    assert abs(rel_err.mean()) < 0.005
    assert rel_err.std() == pytest.approx(0.10, rel=0.03)


def test_no_dead_time_variant() -> None:
    problem = get_problem("furuta", dead_time=False)
    rng = np.random.default_rng(3)
    assert all(sample_instance(problem, rng).extras["dead_time"] == 0.0 for _ in range(50))
    with pytest.raises(ValueError):
        get_problem("cartpole", dead_time=False)


@pytest.mark.parametrize("name", PROBLEM_NAMES)
def test_sample_theta_in_box(name: str) -> None:
    spec = get_problem(name).spec
    rng = np.random.default_rng(4)
    th = np.array([sample_theta(spec, rng) for _ in range(500)])
    assert (th >= spec.theta_lo).all() and (th <= spec.theta_hi).all()


@pytest.mark.parametrize("name", ["boiler", "threetank_pi", "pt2_deadtime", "cartpole"])
def test_screen_keeps_nominal_plant(name: str) -> None:
    problem = get_problem(name)
    assert screen(problem, problem.nominal_instance(), np.random.default_rng(0))


def test_screen_furuta() -> None:
    """The nominal Furuta plant (two control periods of dead time) does not pass the screen, a
    screened prior instance does (the thesis keeps about 1.5 % of Furuta instances)."""
    problem = get_problem("furuta")
    assert not screen(problem, problem.nominal_instance(), np.random.default_rng(0))
    inst = sample_instance(problem, np.random.default_rng(18))
    assert screen(problem, inst, np.random.default_rng(1018))


def test_screen_counts_repeats() -> None:
    """A controller only passes if all repeats succeed: with one repeat more controllers pass."""
    problem = get_problem("pt2_deadtime")
    inst = problem.nominal_instance()
    assert not screen(problem, inst, np.random.default_rng(0), n_theta=100, repeats=10,
                      min_pass=100)
    assert screen(problem, inst, np.random.default_rng(0), n_theta=10, repeats=1, min_pass=1)


def test_reject_probability_formula() -> None:
    assert unsuccessful_reject_probability(0.0) == 0.0
    assert unsuccessful_reject_probability(0.4) == 0.0
    for rho in (0.5, 0.62, 0.98):
        assert unsuccessful_reject_probability(rho) == pytest.approx(
            max(0.0, 1 - (1 - rho) / rho))
        keep = 1 - unsuccessful_reject_probability(rho)
        assert rho * keep / (rho * keep + 1 - rho) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        unsuccessful_reject_probability(1.0)


def test_single_trajectory_prior_caps_unsuccessful_share() -> None:
    problem = get_problem("cartpole")
    sampler = single_trajectory_prior(problem, np.random.default_rng(5), pilot=Pilot(n=300, seed=1))
    assert 0.5 < sampler.rho < 0.75
    assert sampler.reject_prob > 0.2
    n = 400
    bad = sum(not sampler.sample().trajectory.success for _ in range(n))
    assert bad / n <= 0.5 + 2.5 * math.sqrt(0.25 / n)
    assert sampler.n_accepted == n and sampler.n_simulated > n
    # the cached rate is reused, no new pilot
    assert sampler.rho == single_trajectory_prior(problem, np.random.default_rng(6),
                                                  pilot=Pilot(n=300, seed=1)).rho


def test_single_trajectory_prior_skips_pilot_without_failures() -> None:
    sampler = single_trajectory_prior(get_problem("boiler"), np.random.default_rng(0))
    assert sampler.rho == 0.0 and sampler.reject_prob == 0.0
    assert sampler.sample().trajectory.success


def test_trajectory_set_prior_counts() -> None:
    problem = get_problem("cartpole")
    inst = problem.nominal_instance()
    rng = np.random.default_rng(7)
    for n in (1, 7, 30):
        ex = trajectory_set_prior(problem, inst, n, rng, rates=SetRates(0.6, 0.0))
        assert len(ex.trials) == n
        assert ex.n_crashed + ex.n_unsuccessful + ex.n_successful == n
        assert ex.n_unsuccessful == 0
        assert sum(t.traj.crashed for t in ex.trials) == ex.n_crashed
        assert ex.n_simulated >= n
        for t in ex.trials:
            length = t.traj.crash_step or t.T
            assert 1 <= t.t_obs <= length


def test_trajectory_set_prior_cuts_and_pilot() -> None:
    problem = get_problem("pt2_deadtime")
    inst = problem.nominal_instance()
    rng = np.random.default_rng(8)
    full = trajectory_set_prior(problem, inst, 40, rng, cut=False, n_pilot=50)
    assert all(t.t_obs == (t.traj.crash_step or t.T) for t in full.trials)
    cut_any = False
    for _ in range(5):
        ex = trajectory_set_prior(problem, inst, 40, rng, rates=SetRates(0.3, 0.0))
        early = sum(t.early_stopped for t in ex.trials)
        cut_any |= early > 0
        assert 0.0 <= ex.p_cut <= 1.0
    assert cut_any


def test_trajectory_set_prior_unsuccessful_counts() -> None:
    """Furuta has a success criterion: unsuccessful non-crashed trials are counted separately."""
    problem = get_problem("furuta")
    inst = sample_instance(problem, np.random.default_rng(18))
    ex = trajectory_set_prior(problem, inst, 12, np.random.default_rng(9),
                              rates=SetRates(0.5, 0.5))
    kinds = [("c" if t.traj.crashed else "s" if t.traj.success else "u") for t in ex.trials]
    assert kinds.count("c") == ex.n_crashed
    assert kinds.count("u") == ex.n_unsuccessful
    assert kinds.count("s") == ex.n_successful
