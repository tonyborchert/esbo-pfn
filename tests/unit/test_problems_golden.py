"""Parity with the thesis-era Python simulators (references in ``tests/golden``, written by
``tests/golden/make_golden.py`` at the nominal plant with all noise off).

The new simulators repeat the old floating-point operations in the same order, so the logged
channels agree exactly and the cumulative costs agree up to the float32 storage of ``J`` (relative
error below 1e-6, the old cost of some problems is also summed from float32 stage costs). The test
uses a relative bound of 1e-5 on every finite ``J_t`` (tighter than the 1e-4 parity bar) and
requires the same crash step and success flag.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from esbopfn.core import Instance
from esbopfn.problems import PROBLEM_NAMES, get_problem
from esbopfn.problems.pt2_deadtime import sim as pt2_sim

GOLDEN = Path(__file__).resolve().parents[1] / "golden"

#: The old PT2 port appends to its delay line before popping, a delay of 11 base steps (0.11 s).
#: The thesis specifies 0.1 s, which ``simulate`` uses. Parity is checked at the old delay.
OLD_PT2_DEAD_TIME_STEPS = 11


def _cases():
    for name in PROBLEM_NAMES:
        data = np.load(GOLDEN / f"{name}.npz")
        for k in range(int(data["n"])):
            yield pytest.param(name, k, id=f"{name}-{k}")


@pytest.mark.parametrize(("name", "k"), list(_cases()))
def test_matches_old_simulator(name: str, k: int) -> None:
    data = np.load(GOLDEN / f"{name}.npz")
    problem = get_problem(name)
    phi = dict(zip(map(str, data["phi_names"]), data["phi_values"].tolist()))
    extras = dict(zip(map(str, data["extra_names"]), data["extra_values"].tolist()))
    inst = Instance(phi=phi, phi_hat=dict(phi) if problem.prior.uses_phi_hat else {},
                    noise_std={n: 0.0 for n in problem.prior.noise_names}, extras=extras)
    theta = data[f"theta_{k}"]
    rng = np.random.default_rng(int(data[f"seed_{k}"]))
    if name == "pt2_deadtime":
        traj = pt2_sim.rollout(theta, inst, rng, problem.spec.T, OLD_PT2_DEAD_TIME_STEPS)
    else:
        traj = problem.simulate(theta, inst, rng)

    J_old = data[f"J_{k}"]
    crash_old = int(data[f"crash_step_{k}"]) or None
    assert traj.crash_step == crash_old
    assert traj.success == bool(data[f"success_{k}"])
    n_old = len(J_old)
    finite = n_old if crash_old is None else crash_old - 1
    J_new = traj.J[:finite].astype(np.float64)
    np.testing.assert_allclose(J_new, J_old[:finite], rtol=1e-5, atol=0.0)
    if crash_old is not None:
        assert np.isinf(traj.J[crash_old - 1:]).all()
    np.testing.assert_allclose(traj.x[:n_old], data[f"x_{k}"], rtol=0.0, atol=1e-6)
    if traj.xdot is not None:
        np.testing.assert_allclose(traj.xdot[:n_old], data[f"xdot_{k}"], rtol=0.0, atol=1e-6)
