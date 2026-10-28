"""Training of LC-PFN and its variant ``lcpfn-last`` (queries only at the final grid point)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from esbopfn.predictors.lcpfn.model import LCPFN, LCPFNConfig
from esbopfn.problems.registry import get_spec
from esbopfn.training import common, registry
from esbopfn.training.heads import Head, make_head, n_outputs
from esbopfn.training.normalization import Calib
from esbopfn.training.single_trajectory_data import (
    SIZES, VAL_RANGES, LCPFNBatch, LCPFNData, PoolReader, calibrate_lcpfn, cutoff_range,
    training_pools)

log = logging.getLogger(__name__)

VARIANTS = ("lcpfn", "lcpfn-last")


def build(calib: Calib, row: dict[str, Any]) -> LCPFN:
    return LCPFN(LCPFNConfig(n_outputs=n_outputs(calib.head, calib.n_bin),
                             d_model=int(row["d_model"]), n_layers=int(row["n_layers"]),
                             dropout=float(row.get("dropout", 0.0))))


def loss_fn(head: Head):
    def loss(model: torch.nn.Module, batch: LCPFNBatch) -> torch.Tensor:
        if batch.y.shape[0] == 0:   # every row crashed before the cutoff
            return sum(p.sum() for p in model.parameters()) * 0.0
        out = model(batch.t, batch.y_ctx)
        return head.nll(out, batch.y).mean()
    return loss


@torch.no_grad()
def validate(model: torch.nn.Module, head: Head, data: LCPFNData, chunk: int = 1024) -> dict:
    """Validation NLL of the horizon (final grid point) prediction in raw cost units, averaged
    over every (row, cutoff) pair with the row alive at the cutoff, for the ranges of
    :data:`VAL_RANGES`. ``data`` must be built with ``last_only=True``."""
    device = next(model.parameters()).device
    n_cut = len(data.grid) - 1
    sel = {name: cutoff_range(n_cut, lo, hi) for name, (lo, hi) in VAL_RANGES.items()}
    per_k: dict[int, tuple[float, int]] = {}
    for k in sorted(set(np.concatenate(list(sel.values())).tolist())):
        total, count = 0.0, 0
        for a in range(0, data.n_examples, chunk):
            b = data.make(np.arange(a, min(a + chunk, data.n_examples)), int(k))
            if b.y.shape[0] == 0:
                continue
            t, y_ctx, y = b.t.to(device), b.y_ctx.to(device), b.y.to(device)
            with common.autocast_context(device, True):
                out = model(t, y_ctx)
            total += float(head.nll(out, y, raw_units=True).cpu().double().sum())
            count += int(y.numel())
        per_k[int(k)] = (total, count)
    res = {}
    for name, ks in sel.items():
        s = sum(per_k[int(k)][0] for k in ks)
        c = sum(per_k[int(k)][1] for k in ks)
        res[name] = s / max(c, 1)
    return res


def train(problem: str, head: str, size: str, hparams_row: dict[str, Any], seed: int = 0,
          run: str = "0", out_dir: Path | None = None, steps: int | None = None,
          time_limit: float | None = None, device: str = "auto", *, variant: str = "lcpfn",
          train_pool: PoolReader | None = None, val_pool: PoolReader | None = None,
          home: Path | None = None, workers: int = 0, eval_every: int | None = None,
          stat_rows: int | None = None) -> Path:
    """Train one LC-PFN and write its model directory. Same contract as
    :func:`esbopfn.predictors.decpfn.train.train`."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown LC-PFN variant {variant!r}")
    spec = get_spec(problem)
    model_id = registry.ModelId(variant, problem, head, size, str(run))
    out_dir = Path(out_dir) if out_dir is not None else registry.model_dir(model_id, home)
    if registry.is_trained(out_dir):
        log.info("%s is already trained", out_dir)
        return out_dir
    if train_pool is None or val_pool is None:
        pools = training_pools(variant, problem, size, home)
        train_pool, val_pool, n_examples = pools.train, pools.val, pools.n_examples
        datasets = {"train": pools.train_id, "val": pools.val_id}
    else:
        n_examples = min(len(train_pool), SIZES.get(size, len(train_pool)))
        datasets = {"train": str(train_pool.path), "val": str(val_pool.path)}
    out_dir.mkdir(parents=True, exist_ok=True)
    calib_path = out_dir / registry.CALIB_FILE
    if calib_path.exists():
        calib = Calib.from_dict(json.loads(calib_path.read_text()))
    else:
        kw = {"stat_rows": stat_rows} if stat_rows else {}
        calib = calibrate_lcpfn(train_pool, spec, variant, head, seed,
                                n_bin=int(hparams_row.get("n_bin", 1000)), **kw)
        common.write_json(calib_path, calib.to_dict())
    h = make_head(calib)
    batch = int(hparams_row.get("batch_size", 128))
    data = LCPFNData(train_pool, calib, batch, seed, n_examples,
                     last_only=variant == "lcpfn-last",
                     total_steps=steps or n_examples // batch)
    val = LCPFNData(val_pool, calib, batch, seed, None, last_only=True, shuffle=False)
    cfg = common.train_config_from_row(hparams_row, steps=data.steps, seed=seed,
                                       time_limit_s=time_limit, device=device,
                                       eval_every=eval_every)
    model = common.build_model(lambda: build(calib, hparams_row), seed)
    result = common.fit(model, lambda s: data.iterate(s, workers), loss_fn(h), cfg, out_dir,
                        evaluate=lambda m: validate(m, h, val))
    train_cfg = {
        "model_id": model_id.short, "variant": variant, "problem": problem, "head": head,
        "size": size, "seed": int(seed), "run": str(run), "hparams": dict(hparams_row),
        "datasets": datasets, "n_examples": int(n_examples), "steps": int(data.steps),
        "train": cfg.to_dict(), "provenance": "pfn train", "requeue": False,
    }
    metrics = {"result": {"steps_done": result.steps_done, "train_time_s": result.train_time_s,
                          "stopped_by": result.stopped_by, **result.final_metrics}}
    registry.save_model(out_dir, model.state_dict(), {"family": variant, **model.cfg.to_dict()},
                        calib.to_dict(), train_cfg, metrics)
    (out_dir / common.CKPT_NAME).unlink(missing_ok=True)
    return out_dir
