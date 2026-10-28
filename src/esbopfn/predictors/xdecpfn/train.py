"""XDecPFN training on the shared core (:func:`esbopfn.training.common.fit`).

``train(problem, ...)`` reads ``configs/hparams/xdecpfn.yaml``, builds the calibration from the
training set pool, measures the FLOPs of one optimizer step with
``torch.utils.flop_counter.FlopCounterMode``, trains, scores the held-out pool (NLL of the
horizon predictions of cut trajectories and of the unrun queries) and writes the model directory
``xdecpfn/<problem>/default/default/<run>``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from esbopfn.config import configs_dir
from esbopfn.generators.pools import Pool
from esbopfn.predictors.base import resolve_spec
from esbopfn.predictors.xdecpfn.data import SetPoolBatches, XBatch, XCalib, build_calib
from esbopfn.predictors.xdecpfn.model import XDecPFN, XDecPFNConfig, crash_bar_nll
from esbopfn.util.yaml import load_yaml

log = logging.getLogger(__name__)

MODEL = "xdecpfn"
#: Examples per cross-stage chunk (memory bound only, the loss is the same).
CHUNK = 256


# --------------------------------------------------------------------------- shared helpers
def load_hparams(model: str, problem: str, path: Path | str | None = None,
                 overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """The resolved hyperparameters of ``model`` on ``problem``: ``fixed`` plus the per-problem
    step count, then ``overrides``."""
    raw = load_yaml(path or configs_dir() / "hparams" / f"{model}.yaml")
    hp = dict(raw.get("fixed", {}))
    steps = raw.get("steps", {})
    if problem in steps:
        hp["steps"] = int(steps[problem])
    hp["data"] = {k: (v.replace("{p}", problem) if isinstance(v, str) else v)
                  for k, v in raw.get("data", {}).items()}
    hp["thesis_flops"] = raw.get("thesis_flops", {}).get(problem)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(hp.get(key), dict):
            hp[key] = {**hp[key], **value}
        else:
            hp[key] = value
    return hp


def measure_step_flops(model: torch.nn.Module, loss_fn: Callable[[torch.nn.Module, Any], Any],
                       batch: Any) -> int:
    """FLOPs of one forward and backward pass on ``batch`` (the thesis' FLOP measurement)."""
    from torch.utils.flop_counter import FlopCounterMode

    was = model.training
    model.train()
    counter = FlopCounterMode(display=False)
    with counter:
        loss = loss_fn(model, batch)
        loss.backward()
    model.zero_grad(set_to_none=True)
    model.train(was)
    return int(counter.get_total_flops())


def _open_pool(source: Pool | Path | str | None, dataset_id: str,
               home: Path | str | None) -> Pool:
    if isinstance(source, Pool):
        return source
    if source is not None:
        return Pool.open(source)
    from esbopfn.generators import open_dataset

    pool = open_dataset(dataset_id, home)
    if not isinstance(pool, Pool):
        raise TypeError(f"{dataset_id} is not a set pool")
    return pool


# --------------------------------------------------------------------------- loss
def batch_nll(model: XDecPFN, batch: XBatch, log_widths: torch.Tensor,
              chunk: int = CHUNK) -> dict[str, torch.Tensor]:
    """Summed NLL and counts of the horizon predictions (cut trajectories) and unrun queries."""
    h = model.stage1(batch.tokens)
    out = {"traj_sum": h.new_zeros((), dtype=torch.float32), "q_sum": h.new_zeros((), dtype=torch.float32)}
    n_traj = n_q = 0
    E = batch.ex_traj.shape[0]
    for a in range(0, E, chunk):
        b = min(E, a + chunk)
        n = batch.ex_n[a]
        traj, pos = batch.ex_traj[a:b, :n], batch.ex_pos[a:b, :n]
        rep = model.gather(h, traj, pos)
        lt, lq = model.stage2(rep, batch.theta[traj], batch.ex_marker[a:b, :n],
                              batch.ex_valid[a:b, :n], batch.q_theta[a:b])
        scored = batch.ex_scored[a:b, :n]
        nll_t = crash_bar_nll(lt, batch.ex_bin[a:b, :n], batch.ex_crash[a:b, :n], log_widths)
        out["traj_sum"] = out["traj_sum"] + (nll_t * scored).sum()
        n_traj += int(scored.sum())
        nll_q = crash_bar_nll(lq, batch.q_bin[a:b], batch.q_crash[a:b], log_widths)
        out["q_sum"] = out["q_sum"] + nll_q.sum()
        n_q += nll_q.numel()
    out["n_traj"] = torch.tensor(n_traj)
    out["n_q"] = torch.tensor(n_q)
    return out


def make_loss(log_widths: np.ndarray) -> Callable[[XDecPFN, XBatch], torch.Tensor]:
    """Mean of all prediction losses of the batch (thesis M:1131)."""
    cache: dict[torch.device, torch.Tensor] = {}

    def loss_fn(model: XDecPFN, batch: XBatch) -> torch.Tensor:
        dev = batch.tokens.device
        if dev not in cache:
            cache[dev] = torch.as_tensor(log_widths, dtype=torch.float32, device=dev)
        s = batch_nll(model, batch, cache[dev])
        return (s["traj_sum"] + s["q_sum"]) / max(int(s["n_traj"]) + int(s["n_q"]), 1)

    return loss_fn


def make_evaluate(val_batches: list[XBatch], log_widths: np.ndarray):
    """Held-out NLL in ln-J units: horizon predictions, unrun queries and both together."""
    from esbopfn.training.common import to_device

    def evaluate(model: XDecPFN) -> dict[str, float]:
        dev = next(model.parameters()).device
        lw = torch.as_tensor(log_widths, dtype=torch.float32, device=dev)
        st = sq = 0.0
        nt = nq = 0
        for b in val_batches:
            s = batch_nll(model, to_device(b, dev), lw)
            st += float(s["traj_sum"])
            sq += float(s["q_sum"])
            nt += int(s["n_traj"])
            nq += int(s["n_q"])
        return {"val_nll": (st + sq) / max(nt + nq, 1), "val_nll_horizon": st / max(nt, 1),
                "val_nll_unrun": sq / max(nq, 1)}

    return evaluate


# --------------------------------------------------------------------------- train
def model_config(calib: XCalib, d_theta: int, hp: dict[str, Any]) -> XDecPFNConfig:
    return XDecPFNConfig(d_in=calib.d_in, d_theta=d_theta, d_model=int(hp["d_model"]),
                         n_heads=int(hp["n_heads"]), layers_causal=int(hp["layers_causal"]),
                         layers_cross=int(hp["layers_cross"]),
                         d_ff=int(hp.get("d_ff", 2 * hp["d_model"])),
                         dropout=float(hp.get("dropout", 0.0)), n_bins=int(hp["n_bin"]))


def train(problem: str, hparams_row: dict[str, Any] | None = None, seed: int = 0,
          run: str = "0", out_dir: Path | str | None = None, steps: int | None = None,
          time_limit: float | None = None, device: str = "auto", *,
          train_pool: Pool | Path | str | None = None,
          val_pool: Pool | Path | str | None = None, home: Path | str | None = None,
          hparams_file: Path | str | None = None, measure_flops: bool = True) -> Path:
    """Train XDecPFN on ``problem`` and return the model directory.

    ``hparams_row`` overrides entries of the resolved hyperparameters (e.g. a reduced width for
    tests), ``steps`` and ``time_limit`` override the step count and cap the training time.
    ``train_pool`` / ``val_pool`` default to the datasets of the hyperparameter file.
    """
    from esbopfn.training.common import TrainConfig, build_model, fit, resolve_device
    from esbopfn.training.registry import model_dir, save_model

    spec = resolve_spec(problem)
    hp = load_hparams(MODEL, spec.name, hparams_file, hparams_row)
    if steps is not None:
        hp["steps"] = int(steps)
    out = Path(out_dir) if out_dir is not None else model_dir(
        f"{MODEL}/{spec.name}/default/default/{run}", home)
    tpool = _open_pool(train_pool, hp["data"]["train"], home)
    vpool = _open_pool(val_pool, hp["data"]["val"], home)

    calib = build_calib(tpool, spec, int(hp["n_bin"]), int(hp["data"].get("calib_instances", 2000)))
    cfg = model_config(calib, spec.d, hp)
    model = build_model(lambda: XDecPFN(cfg), seed)
    log_widths = calib.log_widths
    loss_fn = make_loss(log_widths)
    stream = SetPoolBatches(tpool, spec, calib, n_sys=int(hp["n_sys"]), n_reuse=int(hp["n_reuse"]),
                            n_unrun=int(hp["n_unrun"]), seed=seed, stream="train")
    val = SetPoolBatches(vpool, spec, calib, n_sys=int(hp["n_sys"]),
                         n_reuse=int(hp["data"].get("val_reuse", 16)),
                         n_unrun=int(hp["n_unrun"]), seed=0, stream="val")
    val_batches = [val.batch(b) for b in range(len(val))]

    tcfg = TrainConfig(lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]),
                       dropout=float(hp.get("dropout", 0.0)),
                       batch_size=int(hp["n_sys"]) * int(hp["n_reuse"]), steps=int(hp["steps"]),
                       warmup_frac=float(hp["warmup_frac"]), clip_grad_norm=hp["clip_grad_norm"],
                       bf16=bool(hp.get("bf16", True)), seed=seed, time_limit_s=time_limit,
                       checkpoint_every=int(hp.get("checkpoint_every", 500)),
                       eval_every=int(hp.get("eval_every", 1000)),
                       log_every=int(hp.get("log_every", 50)), device=device)
    flops = None
    if measure_flops:
        from esbopfn.training.common import to_device

        dev = resolve_device(device)
        model.to(dev)
        flops = measure_step_flops(model, loss_fn, to_device(stream.batch(0), dev))
        log.info("XDecPFN %s: %.3e FLOPs per step, %.3e for %d steps", spec.name, flops,
                 flops * tcfg.steps, tcfg.steps)
    result = fit(model, stream.stream_from, loss_fn, tcfg, out, make_evaluate(val_batches, log_widths))
    n_params = sum(p.numel() for p in model.parameters())
    metrics = {"flops_per_step": flops, "flops_total": None if flops is None else flops * tcfg.steps,
               "flops_thesis": hp.get("thesis_flops"), "n_params": n_params,
               "instances_per_step": int(hp["n_sys"]),
               "instances_consumed": result.steps_done * int(hp["n_sys"]),
               "train_pool_instances": stream.n_instances}
    train_cfg = {"model": MODEL, "problem": spec.name, "variant": "default", "size": "default",
                 "run": str(run), "seed": seed, "hparams": hp, "train": tcfg.to_dict(),
                 "datasets": {"train": str(tpool.path), "val": str(vpool.path)},
                 "provenance": "trained"}
    save_model(out, model.state_dict(), {"class": "XDecPFN", **cfg.to_dict()}, calib.to_dict(),
               train_cfg, metrics)
    return out
