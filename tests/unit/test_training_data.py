"""Single-trajectory batches of DecPFN and LC-PFN."""

from __future__ import annotations

import numpy as np
import pytest

from esbopfn import grids
from esbopfn.core import Trial
from esbopfn.problems.registry import get_spec
from esbopfn.training.single_trajectory_data import (
    DecPFNData, LCPFNData, PoolReader, calibrate_decpfn, calibrate_lcpfn, cutoff_range,
    decpfn_feature_names, decpfn_features, lcpfn_grid, simulate_pool)


@pytest.fixture(scope="module")
def cartpole(tmp_path_factory):
    path = tmp_path_factory.mktemp("pool") / "cartpole"
    simulate_pool(path, "cartpole", 256, seed=3)
    return PoolReader(path)


@pytest.fixture(scope="module")
def boiler(tmp_path_factory):
    path = tmp_path_factory.mktemp("pool") / "boiler"
    simulate_pool(path, "boiler", 64, seed=4)
    return PoolReader(path)


def test_pool_has_crashes(cartpole):
    cs = cartpole.read(np.arange(len(cartpole))).crash_step
    assert 0.1 < (cs > 0).mean() < 0.9


@pytest.mark.parametrize("fixture", ["cartpole", "boiler"])
def test_decpfn_features_match_grid_tokens(fixture, request):
    reader = request.getfixturevalue(fixture)
    spec = get_spec(reader.problem)
    grid = grids.decoder_grid(spec)
    idx = np.arange(20)
    feats, valid = decpfn_features(reader.read(idx), grid, states=True)
    assert feats.shape == (20, len(grid), len(decpfn_feature_names(spec)) - 1)
    for i in idx:
        traj = reader.pool.read(int(i))
        tok = grids.tokens(traj, grid, spec.T)
        n = len(tok)
        assert valid[i].sum() == n and valid[i, :n].all()
        assert np.allclose(feats[i, :n, 0], tok.J, rtol=1e-6)
        assert np.allclose(feats[i, :n, 1:], tok.covariates, rtol=1e-5, atol=1e-5)
        assert not feats[i, n:].any()


def test_decpfn_batches(cartpole):
    spec = get_spec("cartpole")
    calib = calibrate_decpfn(cartpole, spec, "decpfn", "binned", seed=0, n_bin=50)
    assert calib.features == ["J", "dJ", "x_cart", "phi_pole", "sum_xdot_cart",
                              "sum_phidot_pole", "t"]
    assert (calib.time_lo, calib.time_hi) == (1, spec.T - 1) and len(calib.edges) == 51
    data = DecPFNData(cartpole, calib, batch_size=32, seed=0)
    assert data.steps == 256 // 32
    b = data.batch(0)
    L = grids.n_grid(spec) - 1
    assert b.x.shape == (32, L, 7) and b.valid.shape == (32, L) and b.y.shape == (32,)
    crashed = np.isinf(b.y.numpy())
    assert crashed.any() and (~crashed).any()
    # a crashed row is valid strictly before its crash step
    blk = cartpole.read(data.indices(0))
    for i in np.flatnonzero(crashed):
        assert (b.valid[i].numpy() == (calib.grid < blk.crash_step[i])).all()
    assert (b.x[~b.valid] == 0).all()
    # one pass, every row once, seeded order
    seen = np.concatenate([data.indices(s) for s in range(data.steps)])
    assert len(np.unique(seen)) == len(seen) == 256
    again = DecPFNData(cartpole, calib, batch_size=32, seed=0)
    assert np.array_equal(again.indices(3), data.indices(3))
    assert not np.array_equal(DecPFNData(cartpole, calib, 32, seed=1).indices(0), data.indices(0))
    # nostates variant
    cal2 = calibrate_decpfn(cartpole, spec, "decpfn-nostates", "gaussian", 0, states=False)
    assert cal2.features == ["J", "dJ", "t"] and cal2.edges is None
    b2 = DecPFNData(cartpole, cal2, 32, 0, states=False).batch(0)
    assert b2.x.shape == (32, L, 3)


def test_lcpfn_batches_cutoff_and_crash_removal(cartpole):
    spec = get_spec("cartpole")
    calib = calibrate_lcpfn(cartpole, spec, "lcpfn", "linear", seed=0, n_bin=40)
    g = lcpfn_grid(spec)
    assert g[-1] == spec.T and len(g) == grids.n_grid(spec) == 67
    assert (calib.time_lo, calib.time_hi) == (1, spec.T)
    data = LCPFNData(cartpole, calib, batch_size=64, seed=0)
    assert data.kappas.min() >= 1 and data.kappas.max() <= len(g) - 1
    for k in (1, 30, 66):
        idx = data.indices(0)
        b = data.make(idx, k)
        cs = cartpole.read(idx).crash_step
        alive = (cs == 0) | (cs > g[k - 1])
        assert b.y.shape == (alive.sum(), len(g) - k) and b.y_ctx.shape == (alive.sum(), k)
        assert b.t.shape == (alive.sum(), len(g)) and b.kappa == k
        assert np.isfinite(b.y_ctx.numpy()).all()
        # crash targets: inf exactly at query points at or after the crash
        cs_alive = cs[alive]
        q_steps = g[k:]
        expect_inf = (cs_alive[:, None] > 0) & (q_steps[None, :] >= cs_alive[:, None])
        assert np.array_equal(np.isinf(b.y.numpy()), expect_inf)
    last = LCPFNData(cartpole, calib, batch_size=64, seed=0, last_only=True).make(data.indices(0), 10)
    assert last.y.shape[1] == 1 and last.t.shape[1] == 11
    assert np.allclose(last.t[0, -1].item(), calib.normalize_time(np.array([spec.T]))[0])


def test_exact_sizes_give_floor_steps():
    class Fake:
        path = "fake"

        def __len__(self):
            return 1_000_000

    calib = type("C", (), {"grid": np.arange(1, 68)})()
    assert DecPFNData(Fake(), calib, 128, 0, n_examples=100_000).steps == 781
    assert DecPFNData(Fake(), calib, 128, 0, n_examples=1_000_000).steps == 7812
    lc = LCPFNData(Fake(), calib, 128, 0, n_examples=100_000)
    assert lc.steps == 781 and len(lc.kappas) == 781
    with pytest.raises(IndexError):
        lc.indices(781)


def test_cutoff_range():
    k = cutoff_range(66)
    assert k[0] == 14 and k[-1] == 65
    assert cutoff_range(66, 0.0, 0.99)[0] == 1


def test_workers_and_pickling(cartpole):
    import pickle

    spec = get_spec("cartpole")
    calib = calibrate_decpfn(cartpole, spec, "decpfn", "binned", seed=0, n_bin=20, stat_rows=64)
    data = DecPFNData(cartpole, calib, batch_size=64, seed=0)
    clone = pickle.loads(pickle.dumps(data))
    assert np.array_equal(clone.batch(1).x.numpy(), data.batch(1).x.numpy())
    seq = list(data.iterate(1))
    assert len(seq) == data.steps - 1


def test_trial_tokens_for_predictors(cartpole):
    """The predictor path (grids.tokens on a cut trial) sees the same prefix as training."""
    spec = get_spec("cartpole")
    traj = cartpole.pool.read(0)
    t_obs = 200
    tok = grids.tokens(Trial(traj, t_obs).traj, grids.decoder_grid(spec), t_obs)
    assert tok.t.max() <= t_obs
