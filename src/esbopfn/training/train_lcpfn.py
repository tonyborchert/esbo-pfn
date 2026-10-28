"""``pfn train lcpfn|lcpfn-last``: resolve a request and train."""

from __future__ import annotations

from pathlib import Path

from esbopfn.training import TrainRequest, registry


def run(req: TrainRequest) -> Path:
    from esbopfn.predictors.lcpfn.train import train

    head, size = req.head or "binned", req.size or "1M"
    row = registry.hparams_row(req.model, size, head, req.hparams)
    return train(req.problem, head, size, row, seed=req.seed, run=req.run, steps=req.steps,
                 time_limit=req.time_limit, device=req.device, variant=req.model, home=req.home)
