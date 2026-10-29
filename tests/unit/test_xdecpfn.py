"""XDecPFN: tokens, masks, loss, the example law, the set-pool loader, training and prediction."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from esbopfn.core import Trajectory
from esbopfn.grids import decoder_grid
from esbopfn.predictors.base import BarPrediction, get_predictor
from esbopfn.predictors.xdecpfn.data import (
    LN_FLOOR,
    SetPoolBatches,
    XCalib,
    build_calib,
    draw_examples,
    raw_instance,
)
from esbopfn.predictors.xdecpfn.model import (
    MARK_COMPLETE,
    MARK_CRASH,
    MARK_CUT,
    XDecPFN,
    XDecPFNConfig,
    crash_bar_nll,
    cross_mask,
    equal_mass_edges,
)
from esbopfn.problems.registry import get_spec
from esbopfn.training.common import seeded_torch

TINY = {"d_model": 32, "n_heads": 2, "layers_causal": 1, "layers_cross": 1, "d_ff": 64,
        "n_bin": 40, "n_sys": 2, "n_reuse": 16, "log_every": 5, "eval_every": 10,
        "checkpoint_every": 100, "data": {"calib_instances": 4, "val_reuse": 4}}


def synthetic_traj(spec, rng, crash_step=None) -> Trajectory:
    T = spec.T
    J = np.cumsum(rng.uniform(0.5, 1.5, T)).astype(np.float32)
    if crash_step is not None:
        J[crash_step - 1:] = np.inf
    x = rng.normal(size=(T, len(spec.channels))).astype(np.float32)
    xd = rng.normal(size=(T, len(spec.rate_channels))).astype(np.float32) if spec.rate_channels else None
    theta = spec.theta_lo + rng.random(spec.d) * spec.extent
    return Trajectory(theta, J, x, xd, crash_step, crash_step is None)


def tiny_model(spec, d_in, n_bins=40) -> XDecPFN:
    with seeded_torch(0):
        return XDecPFN(XDecPFNConfig(d_in=d_in, d_theta=spec.d, d_model=16, n_heads=2,
                                     layers_causal=1, layers_cross=2, d_ff=32, n_bins=n_bins))


def tiny_calib(spec, n_bins=40) -> XCalib:
    width = 1 + 1 + len(spec.channels) + (len(spec.rate_channels) or len(spec.channels))
    grid = decoder_grid(spec)
    return XCalib(T=spec.T, grid=grid.tolist(), ln_edges=np.linspace(0, 10, n_bins + 1),
                  feat_mean=np.zeros(width), feat_std=np.ones(width), t_mean=500.0, t_std=288.0,
                  theta_mean=(spec.theta_lo + spec.theta_hi) / 2, theta_std=spec.extent / 3.46)


@pytest.fixture
def predictor():
    from esbopfn.predictors.xdecpfn.predictor import XDecPFNPredictor

    spec = get_spec("cartpole")
    cal = tiny_calib(spec)
    return XDecPFNPredictor(tiny_model(spec, cal.d_in), cal, spec)


# --------------------------------------------------------------------------- tokens
def test_token_counts_follow_the_decoder_grid(predictor):
    spec = predictor.spec
    rng = np.random.default_rng(0)
    grid = decoder_grid(spec)
    assert grid.size == 66                                   # N_grid - 1 tokens on Cartpole
    full = synthetic_traj(spec, rng)
    assert predictor._n_obs(full.cut(spec.T)) == 66
    assert predictor._n_obs(full.cut(31)) == 2               # 15, 30
    assert predictor._n_obs(full.cut(14)) == 0
    crashed = synthetic_traj(spec, rng, crash_step=46)       # grid points before the crash: 15 30 45
    assert predictor._n_obs(crashed.cut(spec.T)) == 3
    assert predictor._n_obs(synthetic_traj(spec, rng, crash_step=45).cut(spec.T)) == 2
    assert predictor._token(full.cut(spec.T))[1] == MARK_COMPLETE
    assert predictor._token(full.cut(500))[1] == MARK_CUT
    assert predictor._token(crashed.cut(spec.T))[1] == MARK_CRASH
    assert predictor._token(full.cut(10)) is None            # cut before the first grid point
    assert predictor._token(full.cut(0)) is None


def test_stage1_is_causal():
    spec = get_spec("cartpole")
    m = tiny_model(spec, 7).eval()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 10, 7, generator=g)
    y = x.clone()
    y[:, 6:] = torch.randn(2, 4, 7, generator=g)
    with torch.no_grad():
        assert torch.allclose(m.stage1(x)[:, :6], m.stage1(y)[:, :6], atol=1e-6)


# --------------------------------------------------------------------------- masks
def test_cross_mask_never_reads_queries():
    keys = torch.tensor([[True, True, False, False, False]])   # 2 trajectories, 1 pad, 2 queries
    m = cross_mask(keys)[0, 0]
    assert m.shape == (5, 5)
    off = m & ~torch.eye(5, dtype=torch.bool)
    assert not off[:, 2:].any()                                 # queries and padding are unread
    assert m[:, :2].all()                                       # everyone reads the trajectories
    assert m.diagonal().all()


def test_queries_do_not_change_other_outputs():
    spec = get_spec("cartpole")
    m = tiny_model(spec, 7).eval()
    g = torch.Generator().manual_seed(0)
    rep, th = torch.randn(1, 4, 16, generator=g), torch.randn(1, 4, 4, generator=g)
    mk = torch.tensor([[MARK_COMPLETE, MARK_CUT, MARK_CRASH, MARK_CUT]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    q1 = torch.randn(1, 3, 4, generator=g)
    q2 = q1.clone()
    q2[:, 1:] = torch.randn(1, 2, 4, generator=g)
    with torch.no_grad():
        t1, l1 = m.stage2(rep, th, mk, valid, q1)
        t2, l2 = m.stage2(rep, th, mk, valid, q2)
        t0, _ = m.stage2(rep, th, mk, valid)
    assert torch.allclose(t1, t2, atol=1e-6) and torch.allclose(t1, t0, atol=1e-6)
    assert torch.allclose(l1[:, 0], l2[:, 0], atol=1e-6)
    assert l1.shape == (1, 3, 41) and t1.shape == (1, 4, 41)


# --------------------------------------------------------------------------- loss
def test_crash_bar_nll_two_bins():
    edges = np.array([0.0, 1.0, 3.0])
    logw = torch.tensor(np.log(np.diff(edges)), dtype=torch.float32)
    logits = torch.tensor([[0.3, -0.2, 0.5]] * 3)
    p = np.exp([0.3, -0.2, 0.5])
    p /= p.sum()
    nll = crash_bar_nll(logits, torch.tensor([0, 1, 0]), torch.tensor([False, False, True]), logw)
    want = [-math.log(p[0]) + math.log(1.0), -math.log(p[1]) + math.log(2.0), -math.log(p[2])]
    assert np.allclose(nll.numpy(), want, atol=1e-6)
    # the thesis form -log(1 - w_inf) - log w_k + log width
    w_inf = p[2]
    assert np.isclose(want[1], -math.log(1 - w_inf) - math.log(p[1] / (1 - w_inf)) + math.log(2.0))


def test_equal_mass_edges():
    e = equal_mass_edges(np.r_[np.arange(100.0), np.full(50, 7.0), np.inf], 20)
    assert e.size == 21 and np.all(np.diff(e) > 0) and e[0] == 0 and e[-1] == 99


# --------------------------------------------------------------------------- example law
def test_example_draw_law():
    rng = np.random.default_rng(1)
    N = 180
    n_tok = rng.integers(0, 67, N)
    n_tok[:10] = 0
    crashed = rng.random(N) < 0.3
    crashed[:10] = True
    ex = draw_examples(n_tok, crashed, 3000, 8, 128, rng)
    ns = np.array([t.size for t in ex.traj])
    assert ns.min() >= 1 and ns.max() <= N
    assert abs(ns.mean() - (N + 1) / 2) < 3
    cut_share, rel = [], []
    for idx, pos, mk in zip(ex.traj, ex.pos, ex.marker):
        assert np.unique(idx).size == idx.size
        cut = mk == MARK_CUT
        assert np.all(n_tok[idx][cut] > 0)                     # nothing without a token is cut
        assert np.all(pos[~cut] == n_tok[idx][~cut] - 1)
        assert np.all(mk[~cut] == np.where(crashed[idx][~cut], MARK_CRASH, MARK_COMPLETE))
        cuttable = n_tok[idx] > 0
        if cuttable.any():
            cut_share.append(cut[cuttable].mean())
        rel += list((pos[cut] + 0.5) / n_tok[idx][cut])
    assert abs(np.mean(cut_share) - 0.5) < 0.03                # p ~ U(0, 1)
    assert abs(np.std(cut_share) - 1 / math.sqrt(12)) < 0.04
    assert abs(np.mean(rel) - 0.5) < 0.02                      # uniform cutoff token
    assert ex.labels.shape == (3000, 8)
    assert all(np.unique(row).size == 8 for row in ex.labels)


# --------------------------------------------------------------------------- data and training
@pytest.fixture(scope="module")
def boiler_sets(tmp_path_factory):
    from esbopfn.generators.binaries import BinaryMissing, run

    out = tmp_path_factory.mktemp("sets") / "boiler"
    try:
        run("boiler", "sets", "--out", str(out), "--count", "4", "--seed", "3", "--chunk", "4",
            "--workers", "1", "--resolution", "grid", "--labels", "16")
    except BinaryMissing:
        pytest.skip("plants_gen not built")
    return out


def test_set_pool_loader(boiler_sets):
    from esbopfn.generators.pools import Pool

    spec = get_spec("boiler")
    pool = Pool.open(boiler_sets)
    raw = raw_instance(pool, 0)
    assert raw["feats"].shape == (45, 63, 5)                   # N_max, N_grid - 1, [lnJ dJ x xd t]
    assert np.all(raw["n_tok"] == 63)                          # Boiler cannot crash
    cal = build_calib(pool, spec, n_bins=30, n_instances=4)
    assert cal.ln_edges.size == 31 and np.all(np.diff(cal.ln_edges) > 0)
    assert np.isclose(cal.t_mean, spec.T / 2) and cal.d_in == 5
    b = SetPoolBatches(pool, spec, cal, n_sys=2, n_reuse=5, n_unrun=8, seed=0).batch(0)
    assert b.tokens.shape == (90, 63, 5) and b.ex_traj.shape[0] == 10
    assert list(b.ex_n) == sorted(b.ex_n, reverse=True)
    assert b.q_theta.shape == (10, 8, 1)
    assert int(b.ex_valid.sum()) == sum(b.ex_n)
    assert not torch.any(b.ex_scored & ~b.ex_valid)
    lnJ = np.log(np.maximum(raw["target"], LN_FLOOR))
    assert np.all((lnJ >= cal.ln_edges[0] - 1e-9) & (lnJ <= cal.ln_edges[-1] + 1e-9))


def test_tiny_training_and_prediction(boiler_sets, tmp_path, pfn_home):
    from esbopfn.predictors.xdecpfn.train import train
    from esbopfn.problems import get_problem
    from esbopfn.problems.prior import sample_instance, sample_theta
    from esbopfn.training.registry import ModelNotTrained
    from esbopfn.util.seeds import substream

    with pytest.raises(ModelNotTrained, match="pfn train xdecpfn --problem boiler"):
        get_predictor("xdecpfn/boiler/default/default", "boiler")
    out = train("boiler", TINY, seed=0, steps=20, device="cpu", train_pool=boiler_sets,
                val_pool=boiler_sets)
    assert out == pfn_home / "models/xdecpfn/boiler/default/default/0"
    m = json.loads((out / "metrics.json").read_text())
    losses = [x[1] for x in m["loss"]]
    assert losses[-1] < losses[0]
    assert m["flops_per_step"] > 0 and math.isfinite(m["final"]["val_nll"])
    pr = get_predictor("xdecpfn/boiler/default/default", "boiler")
    P = get_problem("boiler")
    rng = substream(0, "test")
    inst = sample_instance(P, rng)
    tr = [P.simulate(sample_theta(P.spec, rng), inst, rng) for _ in range(4)]
    ctx = [t.cut(t.T) for t in tr[:2]] + [tr[2].cut(100)]
    preds = pr.predict_many([tr[3].cut(c) for c in (0, 5, 50, 300)], ctx)
    for p in preds:
        assert isinstance(p, BarPrediction) and np.all(np.isfinite(p.probs))
        assert np.isclose(p.probs.sum() + p.p_crash, 1.0)
        assert np.allclose(np.log(p.edges), pr.calib.ln_edges)
    assert len(pr.predict_unrun(np.array([[100.0], [300.0]]), ctx)) == 2
    assert np.array_equal(pr.predictable_steps(P.spec), decoder_grid(P.spec))


# --------------------------------------------------------------------------- thesis checkpoint
FURUTA_CKPT = Path("~/Downloads/c1r2_furuta_lnJ").expanduser()


@pytest.mark.skipif(not (FURUTA_CKPT / "f_s20260931_seg6_final.pt").is_file(),
                    reason="thesis XDecPFN checkpoint not present")
def test_legacy_furuta_import(tmp_path):
    from esbopfn.benchmarks.import_models import import_thesis
    from esbopfn.problems import get_problem
    from esbopfn.problems.prior import sample_theta
    from esbopfn.util.seeds import substream

    home = tmp_path / "home"
    (imported,) = import_thesis(FURUTA_CKPT, home=home)
    assert imported.model_id == "xdecpfn/furuta/thesis/default" and imported.step == 16550
    pr = get_predictor("xdecpfn/furuta/thesis/default", "furuta", home=home)
    P = get_problem("furuta")
    rng = substream(5)
    inst = P.nominal_instance(noise=True)
    tr = [P.simulate(sample_theta(P.spec, rng), inst, rng) for _ in range(4)]
    ctx = [t.cut(t.T) for t in tr[:3]]
    p = pr.predict(tr[3].cut(9), ctx)                         # no token yet: unrun query path
    assert isinstance(p, BarPrediction) and np.all(np.isfinite(p.probs))
    u = pr.predict_unrun(np.stack([sample_theta(P.spec, rng) for _ in range(3)]), ctx)
    assert all(0 <= x.p_crash <= 1 and np.isclose(x.probs.sum() + x.p_crash, 1) for x in u)
