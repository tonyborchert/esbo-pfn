"""FT-PFN training on the shared core (:func:`esbopfn.training.common.fit`).

``train(problem, ...)`` reads ``configs/hparams/ftpfn.yaml``, builds the calibration from the
training contexts, measures the FLOPs of one optimizer step, trains in a single pass over
``contexts/<p>/ftpfn`` (32 contexts per step), scores the held-out contexts and writes the model
directory ``ftpfn/<problem>/default/default/<run>``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from esbopfn.generators.pools import ContextStore
from esbopfn.predictors.base import resolve_spec
from esbopfn.predictors.ftpfn.data import (
    QUERY_HORIZON,
    ContextBatches,
    FBatch,
    build_calib,
)
from esbopfn.predictors.ftpfn.model import FTPFN, FTPFNConfig
from esbopfn.predictors.xdecpfn.model import crash_bar_nll
from esbopfn.predictors.xdecpfn.train import load_hparams, measure_step_flops

log = logging.getLogger(__name__)

MODEL = "ftpfn"


def _open_store(source: ContextStore | Path | str | None, dataset_id: str,
                home: Path | str | None) -> ContextStore:
    if isinstance(source, ContextStore):
        return source
    if source is not None:
        return ContextStore(source)
    from esbopfn.generators import open_dataset

    store = open_dataset(dataset_id, home)
    if not isinstance(store, ContextStore):
        raise TypeError(f"{dataset_id} is not a context dataset")
    return store


def query_nll(model: FTPFN, batch: FBatch, log_widths: torch.Tensor) -> torch.Tensor:
    """NLL of every query of the batch ``(Q,)``."""
    logits = model(batch.theta, batch.t, batch.y, batch.kind, batch.ctx, batch.q_row, batch.q_pos)
    return crash_bar_nll(logits, batch.q_bin, batch.q_crash, log_widths)


def make_loss(log_widths: np.ndarray) -> Callable[[FTPFN, FBatch], torch.Tensor]:
    """Mean over each context's queries, then over the contexts (thesis M:1044-1047)."""
    cache: dict[torch.device, torch.Tensor] = {}

    def loss_fn(model: FTPFN, batch: FBatch) -> torch.Tensor:
        dev = batch.theta.device
        if dev not in cache:
            cache[dev] = torch.as_tensor(log_widths, dtype=torch.float32, device=dev)
        nll = query_nll(model, batch, cache[dev])
        E = batch.n_examples
        sums = torch.zeros(E, device=dev).index_add(0, batch.q_example, nll)
        cnt = torch.zeros(E, device=dev).index_add(0, batch.q_example, torch.ones_like(nll))
        has = cnt > 0
        return (sums[has] / cnt[has]).mean()

    return loss_fn


def make_evaluate(val_batches: list[FBatch], log_widths: np.ndarray):
    """Held-out NLL in raw-J units of the horizon queries, the unrun queries and all queries."""
    from esbopfn.training.common import to_device

    def evaluate(model: FTPFN) -> dict[str, float]:
        dev = next(model.parameters()).device
        lw = torch.as_tensor(log_widths, dtype=torch.float32, device=dev)
        sums = {"h": 0.0, "u": 0.0}
        cnts = {"h": 0, "u": 0}
        for b in val_batches:
            b = to_device(b, dev)
            nll = query_nll(model, b, lw)
            hz = b.q_kind == QUERY_HORIZON
            sums["h"] += float(nll[hz].sum())
            sums["u"] += float(nll[~hz].sum())
            cnts["h"] += int(hz.sum())
            cnts["u"] += int((~hz).sum())
        n = max(cnts["h"] + cnts["u"], 1)
        return {"val_nll": (sums["h"] + sums["u"]) / n,
                "val_nll_horizon": sums["h"] / max(cnts["h"], 1),
                "val_nll_unrun": sums["u"] / max(cnts["u"], 1)}

    return evaluate


def model_config(d_theta: int, hp: dict[str, Any]) -> FTPFNConfig:
    return FTPFNConfig(d_theta=d_theta, d_model=int(hp["d_model"]), n_layers=int(hp["n_layers"]),
                       n_heads=int(hp["n_heads"]), d_ff=int(hp.get("d_ff", 2 * hp["d_model"])),
                       dropout=float(hp.get("dropout", 0.0)), n_bins=int(hp["n_bin"]))


def train(problem: str, hparams_row: dict[str, Any] | None = None, seed: int = 0,
          run: str = "0", out_dir: Path | str | None = None, steps: int | None = None,
          time_limit: float | None = None, device: str = "auto", *,
          train_store: ContextStore | Path | str | None = None,
          val_store: ContextStore | Path | str | None = None, home: Path | str | None = None,
          hparams_file: Path | str | None = None, measure_flops: bool = True) -> Path:
    """Train FT-PFN on ``problem`` and return the model directory (see the XDecPFN twin)."""
    from esbopfn.training.common import (
        TrainConfig,
        build_model,
        fit,
        resolve_device,
        to_device,
    )
    from esbopfn.training.registry import model_dir, save_model

    spec = resolve_spec(problem)
    hp = load_hparams(MODEL, spec.name, hparams_file, hparams_row)
    if steps is not None:
        hp["steps"] = int(steps)
    out = Path(out_dir) if out_dir is not None else model_dir(
        f"{MODEL}/{spec.name}/default/default/{run}", home)
    tstore = _open_store(train_store, hp["data"]["train"], home)
    vstore = _open_store(val_store, hp["data"]["val"], home)
    if not tstore.shards or int(tstore.shards[0].meta["T"]) != spec.T:
        raise ValueError(f"{tstore.path} does not hold {spec.name} contexts")

    calib = build_calib(tstore, spec, int(hp["n_bin"]), int(hp["data"].get("calib_examples", 20000)))
    cfg = model_config(spec.d, hp)
    model = build_model(lambda: FTPFN(cfg), seed)
    loss_fn = make_loss(calib.log_widths)
    stream = ContextBatches(tstore, calib, int(hp["batch_size"]))
    val_batches = ContextBatches(vstore, calib, int(hp["batch_size"])).partition()
    tcfg = TrainConfig(lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]),
                       dropout=float(hp.get("dropout", 0.0)), batch_size=int(hp["batch_size"]),
                       steps=int(hp["steps"]), warmup_frac=float(hp["warmup_frac"]),
                       clip_grad_norm=hp["clip_grad_norm"], bf16=bool(hp.get("bf16", True)),
                       seed=seed, time_limit_s=time_limit,
                       checkpoint_every=int(hp.get("checkpoint_every", 2000)),
                       eval_every=int(hp.get("eval_every", 5000)),
                       log_every=int(hp.get("log_every", 100)), device=device)
    flops = None
    if measure_flops:
        dev = resolve_device(device)
        model.to(dev)
        flops = measure_step_flops(model, loss_fn, to_device(stream.batch(0), dev))
        log.info("FT-PFN %s: %.3e FLOPs per step, %.3e for %d steps", spec.name, flops,
                 flops * tcfg.steps, tcfg.steps)
    result = fit(model, stream.stream_from, loss_fn, tcfg, out,
                 make_evaluate(val_batches, calib.log_widths))
    metrics = {"flops_per_step": flops, "flops_total": None if flops is None else flops * tcfg.steps,
               "flops_thesis": hp.get("thesis_flops"),
               "n_params": sum(p.numel() for p in model.parameters()),
               "contexts_consumed": result.steps_done * int(hp["batch_size"]),
               "train_contexts": stream.index.n}
    train_cfg = {"model": MODEL, "problem": spec.name, "variant": "default", "size": "default",
                 "run": str(run), "seed": seed, "hparams": hp, "train": tcfg.to_dict(),
                 "datasets": {"train": str(tstore.path), "val": str(vstore.path)},
                 "provenance": "trained"}
    save_model(out, model.state_dict(), {"class": "FTPFN", **cfg.to_dict()}, calib.to_dict(),
               train_cfg, metrics)
    return out
