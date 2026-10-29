"""Training of DecPFN and its variants ``decpfn-nostates``, ``decpfn-nodeadtime`` and
``decpfn-nodeadtime-nostates``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from esbopfn.predictors.decpfn.model import DecPFN, DecPFNConfig
from esbopfn.problems.registry import get_spec
from esbopfn.training import common, registry
from esbopfn.training.heads import Head, make_head
from esbopfn.training.normalization import Calib
from esbopfn.training.single_trajectory_data import (
    SIZES,
    VAL_RANGES,
    DecPFNBatch,
    DecPFNData,
    PoolReader,
    calibrate_decpfn,
    cutoff_range,
    training_pools,
)

log = logging.getLogger(__name__)

VARIANTS = ("decpfn", "decpfn-nostates", "decpfn-nodeadtime", "decpfn-nodeadtime-nostates")


def uses_states(variant: str) -> bool:
    return not variant.endswith("-nostates")


def build(calib: Calib, row: dict[str, Any]) -> DecPFN:
    from esbopfn.training.heads import n_outputs

    cfg = DecPFNConfig(n_features=calib.n_features, n_outputs=n_outputs(calib.head, calib.n_bin),
                       d_model=int(row["d_model"]), n_layers=int(row["n_layers"]),
                       dropout=float(row.get("dropout", 0.0)))
    return DecPFN(cfg)


def loss_fn(head: Head):
    def loss(model: torch.nn.Module, batch: DecPFNBatch) -> torch.Tensor:
        out = model(batch.x)
        nll = head.nll(out, batch.y[:, None].expand(out.shape[:2]))
        valid = batch.valid.float()
        return (nll * valid).sum() / valid.sum().clamp(min=1.0)
    return loss


@torch.no_grad()
def validate(model: torch.nn.Module, head: Head, data: DecPFNData, chunk: int = 256) -> dict:
    """Validation NLL of the horizon prediction in raw cost units, averaged over every valid
    (row, cutoff) pair whose cutoff lies in the ranges of :data:`VAL_RANGES`."""
    device = next(model.parameters()).device
    L = len(data.grid)
    sel = {name: cutoff_range(L, lo, hi) - 1 for name, (lo, hi) in VAL_RANGES.items()}
    sums = dict.fromkeys(sel, 0.0)
    counts = dict.fromkeys(sel, 0)
    for a in range(0, data.n_examples, chunk):
        b = data.make(np.arange(a, min(a + chunk, data.n_examples)))
        x, valid, y = b.x.to(device), b.valid.to(device), b.y.to(device)
        with common.autocast_context(device, True):
            out = model(x)
        nll = head.nll(out, y[:, None].expand(out.shape[:2]), raw_units=True)
        for name, pos in sel.items():
            m = valid[:, pos]
            sums[name] += float(nll[:, pos][m].cpu().double().sum())
            counts[name] += int(m.sum())
    return {name: sums[name] / max(counts[name], 1) for name in sel}


def train(problem: str, head: str, size: str, hparams_row: dict[str, Any], seed: int = 0,
          run: str = "0", out_dir: Path | None = None, steps: int | None = None,
          time_limit: float | None = None, device: str = "auto", *, variant: str = "decpfn",
          train_pool: PoolReader | None = None, val_pool: PoolReader | None = None,
          home: Path | None = None, workers: int = 0, eval_every: int | None = None,
          stat_rows: int | None = None, rows: int | None = None) -> Path:
    """Train one DecPFN and write its model directory. Returns the directory.

    ``train_pool`` / ``val_pool`` replace the catalog datasets (tests), ``rows`` limits the
    catalog training pool to its first rows (smoke runs). Without ``out_dir`` the
    model goes to ``registry.model_dir(<variant>/<problem>/<head>/<size>/<run>)``. A finished
    directory is left as it is, an unfinished one resumes.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown DecPFN variant {variant!r}")
    spec = get_spec(problem)
    model_id = registry.ModelId(variant, problem, head, size, str(run))
    out_dir = Path(out_dir) if out_dir is not None else registry.model_dir(model_id, home)
    if registry.is_trained(out_dir):
        log.info("%s is already trained", out_dir)
        return out_dir
    if train_pool is None or val_pool is None:
        pools = training_pools(variant, problem, size, home, rows=rows)
        train_pool, val_pool, n_examples = pools.train, pools.val, pools.n_examples
        datasets = {"train": pools.train_id, "val": pools.val_id}
    else:
        n_examples = min(len(train_pool), SIZES.get(size, len(train_pool)))
        datasets = {"train": str(train_pool.path), "val": str(val_pool.path)}
    states = uses_states(variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    calib_path = out_dir / registry.CALIB_FILE
    if calib_path.exists():
        import json

        calib = Calib.from_dict(json.loads(calib_path.read_text()))
    else:
        kw = {"stat_rows": stat_rows} if stat_rows else {}
        calib = calibrate_decpfn(train_pool, spec, variant, head, seed, states,
                                 n_bin=int(hparams_row.get("n_bin", 1000)), **kw)
        common.write_json(calib_path, calib.to_dict())
    h = make_head(calib)
    batch = int(hparams_row.get("batch_size", 128))
    pass_steps = n_examples // batch
    data = DecPFNData(train_pool, calib, batch, seed, n_examples, states,
                      total_steps=steps or pass_steps)
    val = DecPFNData(val_pool, calib, batch, seed, None, states, shuffle=False)
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
        "train": cfg.to_dict(), "provenance": "pfn train",
    }
    metrics = {"result": {"steps_done": result.steps_done, "train_time_s": result.train_time_s,
                          "stopped_by": result.stopped_by, **result.final_metrics}}
    registry.save_model(out_dir, model.state_dict(), {"family": variant, **model.cfg.to_dict()},
                        calib.to_dict(), train_cfg, metrics)
    (out_dir / common.CKPT_NAME).unlink(missing_ok=True)
    return out_dir
