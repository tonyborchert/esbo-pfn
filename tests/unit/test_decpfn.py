"""DecPFN network, training and predictor."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from esbopfn.core import Trial
from esbopfn.predictors.base import BarPrediction, GaussianPrediction, get_predictor
from esbopfn.predictors.decpfn.model import DecPFN, DecPFNConfig
from esbopfn.predictors.decpfn.train import train
from esbopfn.problems.registry import get_spec
from esbopfn.training import registry
from esbopfn.training.single_trajectory_data import PoolReader, simulate_pool

ROW = dict(d_model=32, n_layers=2, lr=2e-3, weight_decay=0.0, dropout=0.0, batch_size=32,
           n_bin=50, warmup_frac=0.25, clip_grad_norm=1.0, bf16=True)


@pytest.fixture(scope="module")
def pools(tmp_path_factory):
    root = tmp_path_factory.mktemp("decpfn")
    simulate_pool(root / "train", "cartpole", 256, seed=11)
    simulate_pool(root / "val", "cartpole", 48, seed=12)
    return PoolReader(root / "train"), PoolReader(root / "val")


def test_forward_shapes_and_causality():
    torch.manual_seed(0)
    m = DecPFN(DecPFNConfig(n_features=6, n_outputs=11, d_model=32, n_layers=2)).eval()
    assert m.cfg.n_heads == 2
    x = torch.randn(3, 9, 7)
    out = m(x)
    assert out.shape == (3, 9, 11)
    x2 = x.clone()
    x2[:, 5:] = torch.randn(3, 4, 7)
    assert torch.allclose(m(x2)[:, :5], out[:, :5], atol=1e-5)
    assert not torch.allclose(m(x2)[:, 5:], out[:, 5:])
    assert m.blocks.layers[0].norm_first and m.blocks.norm is not None


def test_train_reduces_loss_and_predicts(pools, pfn_home):
    train_pool, val_pool = pools
    out = train("cartpole", "binned", "100k", ROW, seed=0, steps=30, device="cpu",
                train_pool=train_pool, val_pool=val_pool, eval_every=10, stat_rows=128)
    assert out == registry.model_dir("decpfn/cartpole/binned/100k")
    m = json.loads((out / "metrics.json").read_text())
    evals = [e["val_nll"] for e in m["evals"]]
    assert evals[-1] < evals[0] and m["steps_done"] == 30
    assert not (out / "ckpt_last.pt").exists()
    calib = json.loads((out / "calib.json").read_text())
    assert calib["head"] == "binned" and len(calib["edges"]) == 51 and calib["time"]["hi"] == 999

    registry.register_predictor_families()
    spec = get_spec("cartpole")
    pred = get_predictor("decpfn/cartpole/binned/100k", spec, device="cpu")
    assert np.array_equal(pred.predictable_steps(spec), np.arange(1, 67) * 15)
    queries = [Trial(val_pool.pool.read(i), t) for i in range(12) for t in (15, 200, 999, 1000)]
    preds = pred.predict_many(queries, [])
    for q, p in zip(queries, preds):
        if q.crashed:
            assert p.p_crash == 1.0
        elif q.complete:
            assert isinstance(p, GaussianPrediction) and p.var == 0.0
        else:
            assert isinstance(p, BarPrediction) and 0 <= p.p_crash <= 1
            assert np.isfinite(p.mean_var()[0])
    # a single prediction equals the batched one (padding does not leak through the mask)
    j = next(j for j, q in enumerate(queries)
             if not q.crashed and not q.complete and q.t_obs == 200)
    q = queries[j]
    single = pred.predict(q, [])
    batched = preds[j]
    assert np.allclose(single.probs, batched.probs, atol=1e-6)
    # predictions only use the tokens up to the largest grid step <= t_obs
    a = pred.predict(Trial(q.traj, 200), [])
    b = pred.predict(Trial(q.traj, 209), [])
    assert np.allclose(a.probs, b.probs, atol=1e-6)
    with pytest.raises(ValueError):
        pred.predict(Trial(q.traj, 3), [])


def test_nostates_gaussian_variant(pools, pfn_home):
    train_pool, val_pool = pools
    out = train("cartpole", "gaussian", "100k", ROW, seed=1, steps=8, device="cpu",
                variant="decpfn-nostates", train_pool=train_pool, val_pool=val_pool,
                stat_rows=64)
    files = registry.load_model(out)
    assert files.model_cfg["n_features"] == 2 and files.calib["features"] == ["J", "dJ", "t"]
    registry.register_predictor_families()
    pred = get_predictor("decpfn-nostates/cartpole/gaussian/100k", "cartpole", device="cpu")
    traj = val_pool.pool.read(int(np.flatnonzero(
        val_pool.read(np.arange(len(val_pool))).crash_step == 0)[0]))
    p = pred.predict(Trial(traj, 300), [])
    assert isinstance(p, GaussianPrediction) and p.var > 0


def test_absent_model_raises(pfn_home):
    registry.register_predictor_families()
    with pytest.raises(registry.ModelNotTrained, match="pfn train decpfn --problem cartpole"):
        get_predictor("decpfn/cartpole/binned/1M", "cartpole")
    with pytest.raises(registry.ModelNotTrained, match="pfn train decpfn-nodeadtime"):
        get_predictor("decpfn-nodeadtime/furuta/binned/1M", "furuta")
    with pytest.raises(ValueError):
        get_predictor("decpfn/boiler/binned/1M", "cartpole")


# ----------------------------------------------------------------------------- nodeadtime-nostates
def test_nodeadtime_nostates_arm_is_wired(pfn_home):
    from esbopfn.cli import build_parser
    from esbopfn.commands.train import requests
    from esbopfn.generators.datasets import load_catalog
    from esbopfn.predictors.base import LAZY_FAMILIES
    from esbopfn.training import TRAINERS, get_trainer

    fam = "decpfn-nodeadtime-nostates"
    assert TRAINERS[fam] == TRAINERS["decpfn-nodeadtime"] and callable(get_trainer(fam))
    assert LAZY_FAMILIES[fam].endswith(":make_nodeadtime_nostates_predictor")
    assert load_catalog().for_model(fam, "furuta") == load_catalog().for_model(
        "decpfn-nodeadtime", "furuta") == ["traj/furuta/1M_nodeadtime", "traj/furuta/val_nodeadtime"]
    assert registry.hparams_base(fam) == "decpfn"
    registry.register_predictor_families()
    with pytest.raises(registry.ModelNotTrained,
                       match=f"pfn train {fam} --problem furuta --head binned --size 1M"):
        get_predictor(f"{fam}/furuta/binned/1M", "furuta")
    reqs = requests(build_parser().parse_args(["train", "all", "--problem", "all"]))
    assert [r.problem for r in reqs if r.model == fam] == ["furuta"]
    with pytest.raises(SystemExit):
        requests(build_parser().parse_args(["train", fam, "--problem", "cartpole"]))


def test_nodeadtime_nostates_trains_and_loads(tmp_path, pfn_home):
    simulate_pool(tmp_path / "train", "furuta", 64, seed=21, dead_time=False)
    simulate_pool(tmp_path / "val", "furuta", 16, seed=22, dead_time=False)
    train_pool, val_pool = PoolReader(tmp_path / "train"), PoolReader(tmp_path / "val")
    fam = "decpfn-nodeadtime-nostates"
    out = train("furuta", "binned", "100k", ROW, seed=0, steps=10, device="cpu", variant=fam,
                train_pool=train_pool, val_pool=val_pool, stat_rows=32)
    assert out == registry.model_dir(f"{fam}/furuta/binned/100k")
    files = registry.load_model(out)
    assert files.model_cfg["family"] == fam
    assert files.model_cfg["n_features"] == 2 and files.calib["features"] == ["J", "dJ", "t"]
    registry.register_predictor_families()
    pred = get_predictor(f"{fam}/furuta/binned/100k", "furuta", device="cpu")
    assert not pred.states and not pred.context_aware
    traj = val_pool.pool.read(0)
    step = int(pred.predictable_steps()[0])
    assert traj.crash_step is None or traj.crash_step > step
    assert pred.features(Trial(traj, step)).shape[1] == 3  # J, dJ, t without the states
    assert isinstance(pred.predict(Trial(traj, step), []), BarPrediction)
