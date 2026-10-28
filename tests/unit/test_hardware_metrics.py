"""Metrics of the hardware figures on hand-made trial lists."""

from __future__ import annotations

import math

import numpy as np
import pytest

from esbopfn.hardware import metrics as M
from esbopfn.hardware import records
from esbopfn.hardware.records import HwRun, HwTrial


def trial(i: int, J: float | None, *, crashed: bool = False, early: bool = False,
          reset: float = 10.0, propose: float = 0.0, control: float = 5.0) -> HwTrial:
    """``J=None`` with ``early`` is an early stop, a finite ``J`` without flags a success."""
    t = 600
    outcome = records.COMPLETE
    if crashed:
        outcome, t = records.CRASHED, 50
    elif early:
        outcome, t = records.EARLY_STOPPED, 100
    x = [[0.0, 0.0]] * t
    return HwTrial(index=i, theta=[0.0] * 8, source="strategy", T=600, t_obs=t,
                   outcome=outcome, crash_step=t if crashed else None,
                   success=outcome == records.COMPLETE,
                   J_obs=math.inf if crashed else float(J if J is not None else 50.0),
                   J=[0.0] * t, x=x, xdot=x, u=None, kick_sign=1.0, incumbent_index=-1,
                   incumbent_J=math.inf,
                   timings={"propose_s": propose, "reset_s": reset, "control_s": control,
                            "total_s": propose + reset + control})


def run(method: str, seed: int, trials: list[HwTrial]) -> HwRun:
    return HwRun("rig", method, seed, trials)


@pytest.fixture
def runs():
    return [
        # reference method: resets of 8 and 12 s -> mean 10 s
        run("decpfn-h", 0, [trial(0, 300.0, reset=8.0), trial(1, 280.0, reset=12.0)]),
        run("esrs", 0, [trial(0, None, crashed=True, reset=20.0), trial(1, 260.0, reset=20.0),
                        trial(2, 270.0, reset=20.0)]),
        run("esrs", 1, [trial(0, None, early=True, reset=1.0), trial(1, 290.0, reset=1.0)]),
        run("bo", 0, [trial(0, 400.0, propose=2.0), trial(1, None, crashed=True, propose=4.0)]),
    ]


def test_reset_normalization(runs):
    assert M.reset_reference(runs, "decpfn-h") == pytest.approx(10.0)
    esrs0 = runs[1]
    # every trial's 20 s reset becomes 10 s: 15 s per trial
    np.testing.assert_allclose(M.wall_axis(esrs0, 10.0), [15.0, 30.0, 45.0])
    np.testing.assert_allclose(M.wall_axis(esrs0), [25.0, 50.0, 75.0])
    assert math.isnan(M.reset_reference(runs, "xdecpfn-pi"))


def test_best_so_far_and_j_best(runs):
    np.testing.assert_array_equal(M.best_so_far(runs[1]), [math.inf, 260.0, 260.0])
    np.testing.assert_array_equal(M.incumbent_indices(runs[1]), [-1, 1, 1])
    best = M.j_best(runs)
    assert (best.J_best, best.method, best.seed) == (260.0, "esrs", 0)
    assert M.j_best(runs, methods=["bo", "decpfn-h"]).J_best == 280.0


def test_regret_axes(runs):
    trials = M.regret_curves(runs, "trials", max_trials=3)
    np.testing.assert_array_equal(trials["x"], [1, 2, 3])
    esrs = trials["methods"]["esrs"]["per_seed"]
    # seed 0: no success, then J_best itself (floor), seed 1: no success, then 290
    assert np.isnan(esrs[0, 0]) and esrs[0, 1] == pytest.approx(-12.0)
    assert np.isnan(esrs[1, 0]) and esrs[1, 1] == pytest.approx(math.log10(30.0))
    assert esrs[1, 2] == pytest.approx(math.log10(30.0))         # forward filled
    dec = trials["methods"]["decpfn-h"]
    assert dec["median"][0] == pytest.approx(math.log10(40.0))
    wall = M.regret_curves(runs, "wall", n_points=5)
    assert wall["reset_ref_s"] == pytest.approx(10.0)
    assert wall["x"][-1] == pytest.approx(45.0)                     # longest normalized run
    bo = wall["methods"]["bo"]["per_seed"][0]
    # bo trial 0 ends at 2 + 10 + 5 = 17 s
    x = wall["x"]
    assert np.all(np.isnan(bo[x < 17.0])) and bo[-1] == pytest.approx(math.log10(140.0))


def test_band_median_and_quantiles():
    m = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, 5.0], [4.0, 7.0]])
    mid, lo, hi = M.band(m)
    np.testing.assert_allclose(mid, [2.5, 6.0])
    np.testing.assert_allclose(lo, [1.75, 5.5])
    np.testing.assert_allclose(hi, [3.25, 6.5])
    mean, _, _ = M.band(np.array([[0.0, 1.0], [2.0, 3.0]]), center="mean")
    np.testing.assert_allclose(mean, [1.0, 2.0])


def test_j_best_seed_excluded(runs):
    out = M.regret_at_fractions(runs, (0.5, 1.0), n_points=10)
    esrs = out["methods"]["esrs"][1.0]
    assert esrs["excluded_seeds"] == [0]
    assert esrs["seeds"] == [1] and esrs["values"] == pytest.approx([math.log10(30.0)])
    assert out["methods"]["bo"][1.0]["values"] == pytest.approx([math.log10(140.0)])


def test_crash_counts_and_curves(runs):
    assert M.crash_counts(runs) == {"decpfn-h": {0: 0}, "esrs": {0: 1, 1: 0}, "bo": {0: 1}}
    cc = M.crash_curves(runs, n_points=10)
    esrs = cc["methods"]["esrs"]
    assert esrs["per_seed"][0, -1] == 1 and esrs["per_seed"][1, -1] == 0
    assert esrs["mean"][-1] == pytest.approx(0.5)
    assert esrs["mean"][0] == 0.0


def test_cost_first_1000s(runs):
    cc = M.cost_curves(runs, t_max=60.0, n_points=7)          # grid 0, 10, ..., 60 s
    esrs = cc["methods"]["esrs"]["per_seed"]
    # seed 0 (normalized 15 s per trial): success 260 at 30 s
    np.testing.assert_array_equal(np.isnan(esrs[0]), [True, True, True, False, False, False,
                                                     False])
    assert esrs[0, 3] == 260.0
    assert cc["methods"]["esrs"]["median"][-1] == pytest.approx(275.0)


def test_incumbent_at(runs):
    inc = M.incumbent_at(runs, "esrs", wall_s=40.0)
    # seed 0 has 260 by 30 s, seed 1 (normalized 15 s per trial) has 290 by 30 s
    assert inc["seed"] == 0 and inc["J"] == 260.0 and inc["trial_index"] == 1
    assert inc["arm_deg"].shape == (600,) and inc["t_s"][-1] == pytest.approx(5.0)
    assert M.incumbent_at(runs, "esrs", wall_s=1.0) is None


def test_unwrapped_pendulum_trace():
    t = trial(0, 270.0)
    t.x = [[0.0, a] for a in np.linspace(3.0, 3.3, 600)]
    t.x = [[0.0, float(math.atan2(math.sin(a), math.cos(a)))] for _, a in t.x]
    inc = M.incumbent_at([run("m", 0, [t])], "m", wall_s=100.0)
    assert np.all(np.diff(inc["pend_deg"]) > 0)                # no jump at +-180 deg


def test_acquisition_wait(runs):
    acq = M.acquisition_wait(runs)
    assert acq["bo"] == {"per_seed": {0: 3.0}, "mean": 3.0}
    assert acq["esrs"]["mean"] == 0.0
