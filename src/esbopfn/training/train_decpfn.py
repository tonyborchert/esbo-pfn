"""``pfn train decpfn|decpfn-nostates|decpfn-nodeadtime|decpfn-nodeadtime-nostates``: resolve a
request and train."""

from __future__ import annotations

from pathlib import Path

from esbopfn.training import TrainRequest, registry


def run(req: TrainRequest) -> Path:
    from esbopfn.predictors.decpfn.train import train

    if req.model.startswith("decpfn-nodeadtime") and req.problem != "furuta":
        raise ValueError(f"{req.model} exists for furuta only")
    head, size = req.head or "binned", req.size or "1M"
    row = registry.hparams_row(req.model, size, head, req.hparams)
    return train(req.problem, head, size, row, seed=req.seed, run=req.run, steps=req.steps,
                 time_limit=req.time_limit, device=req.device, variant=req.model, home=req.home)
