"""Write the golden reference rollouts in this directory from the thesis-era simulators.

Run once with the old prediction code base on the import path, for example::

    PYTHONPATH=<old>/impl/prediction <old>/impl/prediction/.venv/bin/python tests/golden/make_golden.py

Every rollout runs at the nominal plant with all noise switched off. The old simulators still draw
their random start states from ``numpy.random.default_rng(seed)``, which the new simulators reproduce
when they are handed a generator built from the same seed. Each ``<problem>.npz`` holds, per rollout
``k``: ``theta_k``, ``seed_k``, ``J_k`` (cumulative cost up to the last simulated step, float64),
``x_k`` and ``xdot_k`` (channels in the new layout), ``crash_step_k`` (0 for no crash) and
``success_k``. The plant parameters and extras are stored once as ``phi_names``/``phi_values`` and
``extra_names``/``extra_values``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent


def _save(name, phi, extras, rows):
    data = {
        "phi_names": np.array(list(phi)), "phi_values": np.array(list(phi.values()), dtype=np.float64),
        "extra_names": np.array(list(extras), dtype=str),
        "extra_values": np.array(list(extras.values()), dtype=np.float64),
        "n": np.array(len(rows)),
    }
    for k, row in enumerate(rows):
        for key, val in row.items():
            data[f"{key}_{k}"] = np.asarray(val)
    np.savez_compressed(OUT / f"{name}.npz", **data)


def _row(theta, seed, j, x, xdot, crashed, success):
    t = len(j)
    return {"theta": np.asarray(theta, dtype=np.float64), "seed": seed,
            "J": np.cumsum(np.asarray(j, dtype=np.float64)),
            "x": np.asarray(x, dtype=np.float32).reshape(t, -1),
            "xdot": np.zeros((t, 0), np.float32) if xdot is None else np.asarray(xdot, np.float32),
            "crash_step": t if crashed else 0, "success": bool(success)}


def boiler():
    from experiments.boiler.objective import BoilerObjective
    from experiments.boiler.system import DEFAULT_PHI, BoilerBangBangSystem

    sysm, obj = BoilerBangBangSystem(), BoilerObjective()
    rows = []
    for theta in ([20.0], [120.0], [400.0]):
        tr = sysm.recorded_trial(DEFAULT_PHI, np.array(theta), noise_stds=[0.0], seed=0)
        rows.append(_row(theta, 0, obj.calculate_error(tr), np.asarray(tr.Y)[:, 0], None, False, True))
    _save("boiler", {"heating_rate": DEFAULT_PHI[0], "cooling_rate": DEFAULT_PHI[1]}, {}, rows)


def threetank_pi():
    from experiments.threetank_pi.objective import ThreeTankObjective
    from experiments.threetank_pi.system import DEFAULT_PHI, ThreeTankPISystem

    sysm, obj = ThreeTankPISystem(), ThreeTankObjective()
    rows = []
    for theta in ([2.0, 0.05], [0.5, 0.01], [3.0, 0.1]):
        tr = sysm.recorded_trial(DEFAULT_PHI, np.array(theta), noise_stds=[0.0], seed=0)
        rows.append(_row(theta, 0, obj.calculate_error(tr), np.asarray(tr.Y), None, False, True))
    _save("threetank_pi", dict(zip(["V101_opening", "V106_opening", "B101_volume"], DEFAULT_PHI)),
          {}, rows)


def pt2_deadtime():
    from experiments.pt2_deadtime.objective import PT2Objective
    from experiments.pt2_deadtime.system import DEFAULT_PHI, PT2DeadTimePIDSystem

    sysm, obj = PT2DeadTimePIDSystem(), PT2Objective()
    rows = []
    for theta in (np.log10([5.0, 2.0, 1.0]), [0.27, -0.46, -0.92], [0.87, 0.63, -0.99],
                  [-0.75, 0.34, 0.29]):
        tr = sysm.recorded_trial(DEFAULT_PHI, np.array(theta), noise_stds=[0.0], seed=0)
        crashed = bool(np.asarray(tr.failed).any())
        rows.append(_row(theta, 0, obj.calculate_error(tr), np.asarray(tr.Y), None, crashed,
                         not crashed))
    _save("pt2_deadtime", dict(zip(["K", "w0", "D"], DEFAULT_PHI)), {}, rows)


def cartpole():
    from experiments.cartpole import benchmark_goals, default_phi, default_weighting
    from experiments.cartpole.objective import CartPoleCost
    from experiments.cartpole.system import CartPolePIDMujocoSystem

    phi = np.array([1.0, 0.1, 0.5, 10.0], dtype=np.float64)
    assert np.allclose(phi, default_phi)
    sysm, obj = CartPolePIDMujocoSystem(), CartPoleCost(default_weighting)
    rows = []
    for seed, theta in ((0, [2.95, 288.45, 23.39, 1463.05]), (1, [2.7, 252.04, 53.2, 463.7]),
                        (4, [2.57, 48.01, 40.99, 730.9])):
        tr = sysm.recorded_trial(phi, np.array(theta), benchmark_goals, np.zeros(4), seed=seed)
        viz = np.asarray(tr._viz, dtype=np.float32)
        crashed = bool(np.asarray(tr.failed).any())
        rows.append(_row(theta, seed, obj.calculate_error(tr), viz[:, [0, 2]], viz[:, [1, 3]],
                         crashed, not crashed))
    _save("cartpole", dict(zip(["cart_mass", "pole_mass", "pole_half_length", "max_force"], phi)),
          {}, rows)


def furuta():
    from experiments.furuta.objective import FurutaHoldCost
    from experiments.furuta.system import default_phi
    from experiments.furuta.tick_reset import FurutaRealisticTickResetSystem

    dead_time, kick = 2.0 / 120.0, 0.3
    sysm = FurutaRealisticTickResetSystem()
    obj = FurutaHoldCost(sysm.frequency, threshold_deg=5.0)
    phi10 = np.concatenate([default_phi, [dead_time]])
    rows = []
    for seed, theta in enumerate(([60.0, 1.0, 12.0, 5.0, 1.0, 1.0, 1.0, 3.0],
                                  [85.0, 1.15, 11.6, 5.4, 0.6, 1.4, 0.7, 2.0],
                                  [45.0, 0.75, 12.4, 4.6, 1.4, 0.6, 1.3, 1.6],
                                  [70.0, 0.9, 12.2, 4.8, 0.8, 1.2, 1.1, 3.4])):
        tr = sysm.recorded_trial(phi10, np.array(theta), noise_stds=np.zeros(3), phi_hat=phi10,
                                 seed=seed, kick_voltage=kick)
        Y = np.asarray(tr.Y)
        crashed = bool(np.asarray(tr.failed).any())
        success = obj.was_successful(tr)
        rows.append(_row(theta, seed, obj.calculate_error(tr), Y[:, [0, 1]], Y[:, [2, 3]],
                         crashed, success))
    names = ["Rm", "kt", "km", "mr", "Lr", "Dr", "mp", "Lp", "Dp"]
    _save("furuta", dict(zip(names, default_phi)), {"dead_time": dead_time, "kick": kick}, rows)


if __name__ == "__main__":
    for fn in (boiler, threetank_pi, pt2_deadtime, cartpole, furuta):
        fn()
