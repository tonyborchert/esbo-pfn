"""The C++ generators: determinism, worker invariance, pool invariants, screen, set and context
outputs, and parity with the Python simulators of ``esbopfn.problems``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from esbopfn.core import Instance
from esbopfn.generators import binaries
from esbopfn.generators.pools import ContextStore, Pool
from esbopfn.generators.verify import (
    ParityUnavailable,
    check_contexts,
    check_pool,
    parity_deterministic,
    parity_distribution,
)
from esbopfn.grids import curve_grid, decoder_grid
from esbopfn.problems.registry import PROBLEM_NAMES, get_spec
from esbopfn.util.seeds import derive_seed

pytestmark = pytest.mark.cpp

PROBLEMS = list(PROBLEM_NAMES)
#: Problems whose nominal plant passes the degeneracy screen. The nominal Furuta plant with its
#: 16.7 ms dead time does not (about 3 of 100 controllers pass), screened Furuta instances do.
NOMINAL_PASSES = ("boiler", "threetank_pi", "pt2_deadtime", "cartpole")


def _rollout(run, problem: str, *args: str) -> dict:
    return json.loads(run(problem, "rollout", *args))


def test_seed_derivation_matches_python(run):
    for root, label in ((0, 0), (1001, 5), (123456789, 99), (2**40 + 5, 2**33 + 1)):
        assert int(run("cartpole", "derive-seed", "--seed", str(root), "--label", str(label))) == \
            derive_seed(root, label)


def test_sobol_matches_scipy(run):
    from scipy.stats import qmc

    out = run("furuta", "sobol", "--dim", "8", "--n", "16")
    got = np.array([[int(v) for v in line.split()] for line in out.splitlines()], dtype=np.int64)
    ref = np.round(qmc.Sobol(8, scramble=False).random(16) * 2**32).astype(np.int64)
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("problem", PROBLEMS)
def test_info_matches_spec(run, problem):
    info = json.loads(run(problem, "info"))
    spec = get_spec(problem)
    assert info["d"] == spec.d and info["T"] == spec.T
    assert info["theta_names"] == list(spec.theta_names)
    assert info["channels"] == list(spec.channels)
    assert info["rate_channels"] == list(spec.rate_channels)
    np.testing.assert_allclose(info["theta_lo"], spec.theta_lo)
    np.testing.assert_allclose(info["theta_hi"], spec.theta_hi)
    assert info["n_max"] == spec.n_max()
    assert info["decoder_steps_with_horizon"] == list(decoder_grid(spec)) + [spec.T]
    assert info["ftpfn_grid"] == list(curve_grid(spec, 8))


@pytest.mark.parametrize("problem", PROBLEMS)
def test_rollout_deterministic(run, problem):
    a = _rollout(run, problem, "--seed", "7")
    b = _rollout(run, problem, "--seed", "7")
    assert a == b
    c = _rollout(run, problem, "--seed", "8")
    assert c["J"] != a["J"]
    J = np.array(a["J"], dtype=np.float64)
    assert J.shape == (get_spec(problem).T,)
    spec = get_spec(problem)
    mid = ",".join(repr(float(v)) for v in (spec.theta_lo + spec.theta_hi) / 2)
    d = _rollout(run, problem, "--nominal", "--no-noise", "--deterministic", "--seed", "1",
                 "--theta", mid)
    e = _rollout(run, problem, "--nominal", "--no-noise", "--deterministic", "--seed", "2",
                 "--theta", mid)
    assert d["J"] == e["J"] and d["crash_step"] == e["crash_step"]


@pytest.mark.parametrize("problem", PROBLEMS)
def test_worker_count_invariance(run, problem, tmp_path):
    for w in (1, 4):
        run(problem, "traj", "--out", str(tmp_path / f"w{w}"), "--count", "64", "--chunk", "16",
            "--seed", "11", "--workers", str(w), "--rho", "0.6", "--quiet")
    a = (tmp_path / "w1" / "rows.f32").read_bytes()
    b = (tmp_path / "w4" / "rows.f32").read_bytes()
    assert len(a) > 0 and a == b
    # chunks are independent of each other: regenerating chunk 2 alone gives the same rows
    run(problem, "traj", "--out", str(tmp_path / "c2"), "--count", "64", "--chunk", "16",
        "--seed", "11", "--chunks", "2", "--rho", "0.6", "--quiet")
    pool, part = Pool.open(tmp_path / "w1"), Pool.open(tmp_path / "c2")
    np.testing.assert_array_equal(pool.slice(32, 48), part.slice(32, 48))
    assert not (tmp_path / "c2" / "pool.finished").exists()
    assert (tmp_path / "w1" / "pool.finished").exists()


@pytest.mark.parametrize("problem", PROBLEMS)
def test_traj_pool_invariants(run, problem, tmp_path):
    out = tmp_path / "pool"
    run(problem, "traj", "--out", str(out), "--count", "300", "--chunk", "100", "--seed", "3",
        "--workers", "3", "--quiet")
    pool = Pool.open(out)
    rho = json.loads((out / "identity.json").read_text())["rho_unsuccessful"]
    assert len(pool) == 300 and pool.spec.problem == problem
    rep = check_pool(pool, rho_unsuccessful=rho)
    assert rep.ok, rep.lines()
    tr = pool.read(0)
    spec = get_spec(problem)
    assert tr.x.shape == (spec.T, len(spec.channels))
    assert (tr.xdot is None) == (not spec.rate_channels)
    inst = pool.instance(5)
    py = binaries.info(problem)
    assert list(inst.phi) == py["phi_names"] and list(inst.noise_std) == py["noise_names"]
    # resuming a finished pool does nothing, a different seed is refused
    before = (out / "rows.f32").read_bytes()
    run(problem, "traj", "--out", str(out), "--count", "300", "--chunk", "100", "--seed", "3",
        "--quiet", "--rho", repr(rho))
    assert (out / "rows.f32").read_bytes() == before
    with pytest.raises(RuntimeError, match="different dataset"):
        run(problem, "traj", "--out", str(out), "--count", "300", "--chunk", "100", "--seed", "4",
            "--quiet", "--rho", repr(rho))


@pytest.mark.parametrize("problem", PROBLEMS)
def test_screen_nominal_plant(run, problem):
    res = json.loads(run(problem, "screen-one", "--nominal", "--full", "--seed", "0"))
    assert res["n_theta"] == 100 and res["min_pass"] == 25
    if problem in NOMINAL_PASSES:
        assert res["passed"], res
    else:
        assert 0 <= res["passes"] <= 100


@pytest.mark.parametrize("problem", PROBLEMS)
def test_screened_instances(run, problem, tmp_path):
    run(problem, "screen", "--out", str(tmp_path), "--count", "2", "--chunk", "1", "--seed", "5",
        "--workers", "2", "--quiet")
    recs = json.loads((tmp_path / "instances.json").read_text())
    assert [r["index"] for r in recs] == [0, 1]
    for r in recs:
        assert r["screen"]["passes"] >= 25 and r["screen"]["candidates"] >= 1
        assert set(r["instance"]) == {"phi", "phi_hat", "noise_std", "extras", "seed"}


@pytest.mark.parametrize("problem", PROBLEMS)
def test_tiny_sets(run, problem, tmp_path):
    run(problem, "sets", "--out", str(tmp_path), "--count", "2", "--chunk", "1", "--seed", "1",
        "--workers", "2", "--labels", "8", "--quiet")
    spec = get_spec(problem)
    pool = Pool.open(tmp_path)
    assert pool.finished and pool.rows_per_unit == spec.n_max()
    assert len(pool) == 2 * spec.n_max()
    np.testing.assert_array_equal(pool.spec.steps, list(decoder_grid(spec)) + [spec.T])
    rep = check_pool(pool)
    assert rep.ok, rep.lines()
    labels = pool.labels()
    assert len(labels) == 16 and labels.rows_per_unit == 8
    assert check_pool(labels).ok
    g = pool.read_grid(3)
    assert g.instance == 0 and g.x.shape == (len(g.steps), len(spec.channels))
    inst = pool.instance(1)
    assert isinstance(inst, Instance) and inst.seed > 0


@pytest.mark.parametrize("problem", PROBLEMS)
def test_tiny_contexts(run, problem, tmp_path):
    run(problem, "contexts", "--out", str(tmp_path), "--count", "4", "--chunk", "2", "--seed", "1",
        "--workers", "2", "--quiet")
    store = ContextStore(tmp_path)
    spec = get_spec(problem)
    assert len(store) == 4 and (tmp_path / "pool.finished").exists()
    for ex in store:
        assert 1 <= ex.n <= spec.n_max()
        assert ex.unrun_theta.shape == (16, spec.d)
        assert np.all(ex.tok_t <= spec.T) and np.all(ex.length >= 1)
    rep = check_contexts(store)
    assert rep.ok, rep.lines()


@pytest.mark.parametrize("problem", PROBLEMS)
def test_parity_with_python(problem, tmp_path, run):
    try:
        cases = parity_deterministic(problem, n_theta=6)
    except ParityUnavailable as exc:
        pytest.skip(f"Python simulators unavailable: {exc}")
    bad = [c for c in cases if not c.ok]
    assert not bad, bad
    if problem == "furuta":
        # the nominal plant crashes for most controllers: also compare on a screened instance
        run(problem, "screen", "--out", str(tmp_path), "--count", "1", "--chunk", "1", "--seed", "2",
            "--quiet")
        rec = json.loads((tmp_path / "instances.json").read_text())[0]["instance"]
        inst = Instance(phi=rec["phi"], phi_hat=rec["phi_hat"], noise_std=rec["noise_std"],
                        extras=rec["extras"], seed=rec["seed"])
        cases = parity_deterministic(problem, n_theta=12, instance=inst)
        assert all(c.ok for c in cases), [c for c in cases if not c.ok]
        assert any(c.crash_cpp is None for c in cases)
    dist = parity_distribution(problem, n=300, seed=1)
    assert dist.ok(), dist
