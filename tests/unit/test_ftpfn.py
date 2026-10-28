"""FT-PFN: tokens, the block mask, the loss reduction, the shard loader, training and prediction."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from esbopfn.core import Trajectory
from esbopfn.grids import curve_grid
from esbopfn.predictors.base import BarPrediction, get_predictor
from esbopfn.predictors.ftpfn.data import (
    QUERY_HORIZON,
    QUERY_UNRUN,
    ContextBatches,
    ContextTokens,
    FCalib,
    build_calib,
    example_tokens,
    pack,
    trial_tokens,
)
from esbopfn.predictors.ftpfn.model import (
    FTPFN,
    KIND_CRASH,
    KIND_QUERY,
    FTPFNConfig,
    block_mask,
)
from esbopfn.predictors.ftpfn.train import make_loss
from esbopfn.problems.registry import get_spec

TINY = {"d_model": 32, "n_heads": 2, "n_layers": 2, "d_ff": 64, "n_bin": 40, "batch_size": 4,
        "log_every": 5, "eval_every": 10, "checkpoint_every": 100, "data": {"calib_examples": 16}}


def synthetic_traj(spec, rng, crash_step=None) -> Trajectory:
    J = np.cumsum(rng.uniform(0.5, 1.5, spec.T)).astype(np.float32)
    if crash_step is not None:
        J[crash_step - 1:] = np.inf
    x = rng.normal(size=(spec.T, len(spec.channels))).astype(np.float32)
    theta = spec.theta_lo + rng.random(spec.d) * spec.extent
    return Trajectory(theta, J, x, None, crash_step, crash_step is None)


def toy_calib(spec, n_bins=10) -> FCalib:
    return FCalib(T=spec.T, grid=curve_grid(spec, 8).tolist(), edges=np.linspace(0, 1500, n_bins + 1),
                  y_mean=300.0, y_std=200.0, y_lo=0.0, y_hi=1500.0, t_mean=(spec.T + 1) / 2,
                  t_std=spec.T / 3.46, theta_mean=(spec.theta_lo + spec.theta_hi) / 2,
                  theta_std=spec.extent / 3.46)


def tiny_model(d_theta, n_bins=10) -> FTPFN:
    torch.manual_seed(0)
    return FTPFN(FTPFNConfig(d_theta=d_theta, d_model=16, n_layers=2, n_heads=2, d_ff=32,
                             n_bins=n_bins)).eval()


# --------------------------------------------------------------------------- tokens
def test_trial_tokens():
    spec = get_spec("pt2_deadtime")                      # grid 62 125 187 250 313 375 438 501
    grid = curve_grid(spec, 8)
    rng = np.random.default_rng(0)
    full = synthetic_traj(spec, rng)
    t, J = trial_tokens(full.cut(spec.T), grid)
    assert np.array_equal(t, grid)                       # complete: 8 tokens, no extra
    t, _ = trial_tokens(full.cut(200), grid)
    assert t.tolist() == [62, 125, 187, 200]             # truncated: last observed step added
    t, _ = trial_tokens(full.cut(125), grid)
    assert t.tolist() == [62, 125, 125]                  # the extra token even on a grid step
    crashed = synthetic_traj(spec, rng, crash_step=92)
    t, J = trial_tokens(crashed.cut(spec.T), grid)
    assert t.tolist() == [62, 92] and np.isfinite(J[0]) and np.isinf(J[1])
    assert trial_tokens(full.cut(0), grid)[0].size == 0
    t, _ = trial_tokens(full.cut(30), grid)
    assert t.tolist() == [30]


# --------------------------------------------------------------------------- packing and mask
def _ctx(n_tok, n_q, d=1, crash_last=False) -> ContextTokens:
    J = np.linspace(10, 100, n_tok)
    if crash_last:
        J[-1] = np.inf
    return ContextTokens(theta=np.full((n_tok, d), 100.0), t=np.linspace(50, 700, n_tok), J=J,
                         q_theta=np.full((n_q, d), 200.0), q_target=np.full(n_q, 500.0),
                         q_kind=np.r_[np.full(1, QUERY_HORIZON), np.full(n_q - 1, QUERY_UNRUN)])


def test_pack_and_block_mask():
    spec = get_spec("boiler")
    cal = toy_calib(spec)
    b = pack([_ctx(10, 3), _ctx(4, 2, crash_last=True), _ctx(3, 2)], cal)
    assert b.theta.shape[0] == 2 and b.theta.shape[1] == 13      # 13 + (6 + 5) fits two rows
    assert int((b.kind == KIND_QUERY).sum()) == 7 and int((b.kind == KIND_CRASH).sum()) == 1
    assert b.q_example.tolist() == [0, 0, 0, 1, 1, 2, 2]
    m = block_mask(b.kind, b.ctx)[:, 0]
    eye = torch.eye(m.shape[-1], dtype=torch.bool)
    off = m & ~eye
    assert not off[(b.kind == KIND_QUERY)[:, None, :].expand_as(off)].any()   # nobody reads queries
    same = b.ctx[:, :, None] == b.ctx[:, None, :]
    assert not (off & ~same).any()                                           # contexts are separate
    assert m.diagonal(dim1=1, dim2=2).all()


def test_queries_and_neighbours_are_independent():
    spec = get_spec("boiler")
    cal = toy_calib(spec)
    model = tiny_model(1)
    a = pack([_ctx(6, 3), _ctx(5, 2)], cal)
    other = _ctx(5, 2)
    other.J = other.J * 3
    c = _ctx(6, 3)
    c.q_theta[1:] = 400.0
    b = pack([c, other], cal)
    with torch.no_grad():
        la = model(a.theta, a.t, a.y, a.kind, a.ctx, a.q_row, a.q_pos)
        lb = model(b.theta, b.t, b.y, b.kind, b.ctx, b.q_row, b.q_pos)
    assert torch.allclose(la[0], lb[0], atol=1e-5)          # first query of context 0 unchanged


def test_loss_is_mean_within_then_over_examples():
    spec = get_spec("boiler")
    cal = toy_calib(spec)
    model = tiny_model(1)
    b = pack([_ctx(6, 4), _ctx(5, 2)], cal)
    loss = make_loss(cal.log_widths)(model, b)
    from esbopfn.predictors.ftpfn.train import query_nll

    with torch.no_grad():
        nll = query_nll(model, b, torch.tensor(cal.log_widths, dtype=torch.float32))
    want = 0.5 * (nll[:4].mean() + nll[4:].mean())
    assert torch.isclose(loss.detach(), want) and math.isfinite(float(loss.detach()))


# --------------------------------------------------------------------------- shards and training
@pytest.fixture(scope="module")
def boiler_contexts(tmp_path_factory):
    from esbopfn.generators.binaries import BinaryMissing, run

    out = tmp_path_factory.mktemp("ctx") / "boiler"
    try:
        run("boiler", "contexts", "--out", str(out), "--count", "16", "--seed", "3", "--chunk",
            "8", "--workers", "1", "--n-unrun", "16")
    except BinaryMissing:
        pytest.skip("plants_gen not built")
    return out


def test_context_loader(boiler_contexts):
    from esbopfn.generators.pools import STATUS_TRUNCATED, ContextStore

    spec = get_spec("boiler")
    store = ContextStore(boiler_contexts)
    assert len(store) == 16
    cal = build_calib(store, spec, n_bins=20, n_examples=16)
    assert cal.edges.size == 21 and cal.grid == curve_grid(spec, 8).tolist()
    ex = store.example(9)
    tok = example_tokens(ex)
    assert tok.q_kind.size == int((ex.status == STATUS_TRUNCATED).sum()) + 16
    assert tok.t.size == int((ex.tok_t > 0).sum())
    cb = ContextBatches(store, cal, batch_size=4)
    b = cb.batch(1)
    assert b.n_examples == 4 and int(b.q_valid.sum()) == b.q_row.numel()
    assert sum(len(x.q_kind) for x in [example_tokens(store.example(i)) for i in range(4, 8)]) \
        == b.q_row.numel()
    assert len(cb.partition()) == 4


def test_tiny_training_and_prediction(boiler_contexts, pfn_home):
    from esbopfn.predictors.ftpfn.train import train
    from esbopfn.problems import get_problem
    from esbopfn.problems.prior import sample_instance, sample_theta
    from esbopfn.training.registry import ModelNotTrained
    from esbopfn.util.seeds import substream

    with pytest.raises(ModelNotTrained, match="pfn train ftpfn --problem boiler"):
        get_predictor("ftpfn/boiler/default/default", "boiler")
    out = train("boiler", TINY, steps=20, device="cpu", train_store=boiler_contexts,
                val_store=boiler_contexts)
    m = json.loads((out / "metrics.json").read_text())
    losses = [x[1] for x in m["loss"]]
    assert losses[-1] < losses[0] and m["flops_per_step"] > 0
    assert math.isfinite(m["final"]["val_nll_horizon"]) and math.isfinite(m["final"]["val_nll_unrun"])
    pr = get_predictor("ftpfn/boiler/default/default", "boiler")
    P = get_problem("boiler")
    rng = substream(0, "test")
    inst = sample_instance(P, rng)
    tr = [P.simulate(sample_theta(P.spec, rng), inst, rng) for _ in range(4)]
    ctx = [t.cut(t.T) for t in tr[:2]] + [tr[2].cut(100)]
    for p in pr.predict_many([tr[3].cut(c) for c in (0, 5, 300)], ctx) + \
            pr.predict_unrun(np.array([[50.0], [250.0]]), ctx):
        assert isinstance(p, BarPrediction) and np.isclose(p.probs.sum() + p.p_crash, 1.0)
    assert pr.predictable_steps(P.spec).tolist() == curve_grid(P.spec, 8)[:-1].tolist()


BOILER_CKPT = Path("~/Downloads/dynformer_timing_bundle/checkpoints/fr2/boiler/final.pt").expanduser()


@pytest.mark.skipif(not BOILER_CKPT.is_file(), reason="thesis FT-PFN checkpoint not present")
def test_legacy_boiler_import(tmp_path):
    from esbopfn.benchmarks.import_models import import_ftpfn_checkpoint
    from esbopfn.problems import get_problem
    from esbopfn.problems.prior import sample_instance, sample_theta
    from esbopfn.util.seeds import substream

    home = tmp_path / "home"
    imp = import_ftpfn_checkpoint(BOILER_CKPT, home=home)
    assert imp.model_id == "ftpfn/boiler/thesis/default" and imp.step == 109375
    pr = get_predictor("ftpfn/boiler/thesis/default", "boiler", home=home)
    P = get_problem("boiler")
    rng = substream(2)
    inst = sample_instance(P, rng)
    tr = [P.simulate(sample_theta(P.spec, rng), inst, rng) for _ in range(5)]
    p = pr.predict(tr[4].cut(300), [t.cut(t.T) for t in tr[:4]])
    assert isinstance(p, BarPrediction) and np.all(np.isfinite(p.probs))
    mean, _ = p.mean_var()
    assert abs(mean - tr[4].J[-1]) / tr[4].J[-1] < 0.5
