"""A few hundred training steps of XDecPFN and FT-PFN at reduced width on the GPU (CUDA or MPS)."""

from __future__ import annotations

import json
import math

import pytest
import torch

pytestmark = pytest.mark.gpu


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    pytest.skip("no GPU")


def _gen(problem: str, sub: str, out, *args: str):
    from esbopfn.generators.binaries import BinaryMissing, run

    try:
        run(problem, sub, "--out", str(out), "--seed", "11", "--workers", "4", *args)
    except BinaryMissing:
        pytest.skip("generator binaries not built")
    return out


def _loss_drop(out) -> tuple[float, float, dict]:
    m = json.loads((out / "metrics.json").read_text())
    losses = [x[1] for x in m["loss"]]
    return losses[0], sum(losses[-3:]) / 3, m


def test_xdecpfn_gpu_steps(tmp_path, pfn_home):
    from esbopfn.predictors.xdecpfn.train import train

    train_pool = _gen("cartpole", "sets", tmp_path / "sets", "--count", "24", "--chunk", "8",
                      "--resolution", "grid", "--labels", "128")
    val_pool = _gen("cartpole", "sets", tmp_path / "val", "--count", "4", "--chunk", "4",
                    "--resolution", "grid", "--labels", "128")
    hp = {"d_model": 64, "n_heads": 4, "layers_causal": 2, "layers_cross": 2, "d_ff": 128,
          "n_sys": 4, "n_reuse": 64, "log_every": 20, "eval_every": 100,
          "checkpoint_every": 1000, "data": {"calib_instances": 24, "val_reuse": 8}}
    out = train("cartpole", hp, steps=300, device=_device(), train_pool=train_pool,
                val_pool=val_pool)
    first, last, m = _loss_drop(out)
    assert last < first - 0.3
    assert math.isfinite(m["final"]["val_nll"]) and m["steps_done"] == 300


def test_ftpfn_gpu_steps(tmp_path, pfn_home):
    from esbopfn.predictors.ftpfn.train import train

    train_store = _gen("cartpole", "contexts", tmp_path / "ctx", "--count", "512", "--chunk",
                       "128", "--n-unrun", "16")
    val_store = _gen("cartpole", "contexts", tmp_path / "val", "--count", "32", "--chunk", "32",
                     "--n-unrun", "16")
    hp = {"d_model": 64, "n_heads": 4, "n_layers": 3, "d_ff": 128, "batch_size": 8, "lr": 3e-4,
          "log_every": 20, "eval_every": 100, "checkpoint_every": 1000,
          "data": {"calib_examples": 512}}
    out = train("cartpole", hp, steps=300, device=_device(), train_store=train_store,
                val_store=val_store)
    first, last, m = _loss_drop(out)
    assert last < first - 0.3
    assert math.isfinite(m["final"]["val_nll"]) and m["steps_done"] == 300
