"""Short real trainings of LC-PFN and DecPFN on the accelerator (CUDA or MPS)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from esbopfn.core import Trial
from esbopfn.predictors.base import BarPrediction, conditioned_log_likelihood, get_predictor
from esbopfn.training import registry
from esbopfn.training.single_trajectory_data import PoolReader, simulate_pool

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _opt_in(request):
    """Real trainings take seconds per model, so they run only with --run-gpu."""
    if not request.config.getoption("--run-gpu"):
        pytest.skip("needs --run-gpu")

ROW = dict(d_model=64, n_layers=2, lr=1e-3, weight_decay=1e-4, dropout=0.0, batch_size=32,
           n_bin=1000, warmup_frac=0.25, clip_grad_norm=1.0, bf16=True)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "mps"


@pytest.fixture(scope="module")
def pools(tmp_path_factory):
    root = tmp_path_factory.mktemp("gpu")
    simulate_pool(root / "train", "cartpole", 2000, seed=31)
    simulate_pool(root / "val", "cartpole", 128, seed=32)
    return PoolReader(root / "train"), PoolReader(root / "val")


@pytest.mark.parametrize("model", ["decpfn", "lcpfn"])
def test_short_training_on_device(model, pools, pfn_home):
    if model == "decpfn":
        from esbopfn.predictors.decpfn.train import train
    else:
        from esbopfn.predictors.lcpfn.train import train
    train_pool, val_pool = pools
    out = train("cartpole", "binned", "100k", ROW, seed=0, steps=300, device=_device(),
                train_pool=train_pool, val_pool=val_pool, eval_every=100)
    m = json.loads((out / "metrics.json").read_text())
    losses = [v for _, v, _ in m["loss"]]
    assert m["steps_done"] == 300 and losses[-1] < losses[0]
    evals = [e["val_nll"] for e in m["evals"]]
    assert evals[-1] < evals[0]
    registry.register_predictor_families()
    pred = get_predictor(f"{model}/cartpole/binned/100k", "cartpole", device=_device())
    alive = np.flatnonzero(val_pool.read(np.arange(len(val_pool))).crash_step == 0)
    queries = [Trial(val_pool.pool.read(int(i)), 450) for i in alive[:16]]
    preds = pred.predict_many(queries, [])
    assert all(isinstance(p, BarPrediction) and np.isfinite(p.mean_var()[0]) for p in preds)
    ll = [conditioned_log_likelihood(p, float(q.traj.J[-1]), q.J_obs) for p, q in zip(preds, queries)]
    assert np.isfinite(ll).all()
