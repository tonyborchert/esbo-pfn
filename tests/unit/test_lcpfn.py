"""LC-PFN network, training and predictor."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from esbopfn.core import Trial
from esbopfn.predictors.base import BarPrediction, GaussianPrediction, get_predictor
from esbopfn.predictors.lcpfn.model import LCPFN, LCPFNConfig, pfn_mask
from esbopfn.predictors.lcpfn.train import train
from esbopfn.problems.registry import get_spec
from esbopfn.training import registry
from esbopfn.training.common import seeded_torch
from esbopfn.training.single_trajectory_data import PoolReader, simulate_pool

ROW = dict(d_model=32, n_layers=2, lr=2e-3, weight_decay=0.0, dropout=0.0, batch_size=32,
           n_bin=50, warmup_frac=0.25, clip_grad_norm=1.0, bf16=True)


@pytest.fixture(scope="module")
def pools(tmp_path_factory):
    root = tmp_path_factory.mktemp("lcpfn")
    simulate_pool(root / "train", "cartpole", 256, seed=21)
    simulate_pool(root / "val", "cartpole", 32, seed=22)
    return PoolReader(root / "train"), PoolReader(root / "val")


def test_mask_and_forward():
    mask = pfn_mask(2, 4)
    assert mask.tolist() == [[False, False, True, True], [False, False, True, True],
                             [False, False, False, True], [False, False, True, False]]
    with seeded_torch(0):
        m = LCPFN(LCPFNConfig(n_outputs=7, d_model=32, n_layers=2)).eval()
    g = torch.Generator().manual_seed(0)
    blk = m.blocks[0]
    assert not blk.norm_first and torch.count_nonzero(blk.linear2.weight) == 0
    assert torch.count_nonzero(blk.self_attn.out_proj.weight) == 0
    assert isinstance(m.decoder[1], torch.nn.GELU) and m.decoder[0].out_features == 64
    t, y = torch.randn(3, 10, generator=g), torch.randn(3, 4, generator=g)
    out = m(t, y)
    assert out.shape == (3, 6, 7)
    # queries do not see each other: dropping later queries leaves earlier outputs unchanged
    for p in m.parameters():   # break the zero init so attention matters
        p.data.add_(0.05 * torch.randn(p.shape, generator=g))
    full = m(t, y)
    assert torch.allclose(m(t[:, :6], y)[:, :2], full[:, :2], atol=1e-5)


def test_train_reduces_loss_and_predicts(pools, pfn_home):
    train_pool, val_pool = pools
    out = train("cartpole", "linear", "100k", ROW, seed=0, steps=30, device="cpu",
                train_pool=train_pool, val_pool=val_pool, eval_every=10, stat_rows=128)
    m = json.loads((out / "metrics.json").read_text())
    evals = [e["val_nll"] for e in m["evals"]]
    assert evals[-1] < evals[0]
    spec = get_spec("cartpole")
    pred = get_predictor("lcpfn/cartpole/linear/100k", spec, device="cpu")
    grid = np.asarray(json.loads((out / "calib.json").read_text())["grid"])
    assert np.array_equal(pred.predictable_steps(spec), grid[:-1]) and grid[-1] == spec.T
    alive = np.flatnonzero(val_pool.read(np.arange(len(val_pool))).crash_step == 0)
    traj = val_pool.pool.read(int(alive[0]))
    p = pred.predict(Trial(traj, 500), [])
    assert isinstance(p, BarPrediction) and np.isfinite(p.mean_var()[0])
    assert np.allclose(pred.predict(Trial(traj, 500), []).probs,
                       pred.predict(Trial(traj, grid[grid <= 500][-1]), []).probs, atol=1e-6)
    curve = pred.predict_curve(Trial(traj, 500))
    assert np.array_equal(curve.steps, grid[grid > 500]) and len(curve.predictions) == len(curve.steps)
    assert np.allclose(curve.predictions[-1].probs, p.probs, atol=1e-5)
    crashed = Trial(val_pool.pool.read(int(np.flatnonzero(
        val_pool.read(np.arange(len(val_pool))).crash_step > 0)[0])), 1000)
    assert pred.predict(crashed, []).p_crash == 1.0
    assert all(c.p_crash == 1.0 for c in pred.predict_curve(crashed).predictions)


def test_last_variant_gaussian(pools, pfn_home):
    train_pool, val_pool = pools
    out = train("cartpole", "gaussian", "100k", ROW, seed=0, steps=8, device="cpu",
                variant="lcpfn-last", train_pool=train_pool, val_pool=val_pool, stat_rows=64)
    assert registry.load_model(out).train_cfg["variant"] == "lcpfn-last"
    registry.register_predictor_families()
    pred = get_predictor("lcpfn-last/cartpole/gaussian/100k", "cartpole", device="cpu")
    traj = val_pool.pool.read(0)
    assert isinstance(pred.predict(Trial(traj, 100), []), GaussianPrediction)


def test_absent_model_raises(pfn_home):
    registry.register_predictor_families()
    with pytest.raises(registry.ModelNotTrained, match="pfn train lcpfn-last --problem boiler"):
        get_predictor("lcpfn-last/boiler/binned/1M", "boiler")
