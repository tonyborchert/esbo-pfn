"""``pfn train xdecpfn``: thin wrapper around :func:`esbopfn.predictors.xdecpfn.train.train`.

``run(request)`` is the :data:`esbopfn.training.TRAINERS` entry
(``"xdecpfn": "esbopfn.training.train_xdecpfn:run"``), :func:`register` adds it to a trainer table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from esbopfn.predictors.xdecpfn.train import train

__all__ = ["register", "run", "train"]

ENTRY = "esbopfn.training.train_xdecpfn:run"


def run(request: Any) -> Path:
    """Train XDecPFN for one :class:`~esbopfn.training.TrainRequest` (head and size are ignored,
    XDecPFN has one configuration)."""
    return train(request.problem, None, seed=request.seed, run=request.run,
                 steps=request.steps, time_limit=request.time_limit, device=request.device,
                 home=request.home, hparams_file=request.hparams)


def register(trainers: dict[str, Any] | None = None) -> dict[str, Any]:
    """Add ``"xdecpfn"`` to ``trainers`` (default :data:`esbopfn.training.TRAINERS`)."""
    if trainers is None:
        from esbopfn.training import TRAINERS as trainers
    trainers.setdefault("xdecpfn", ENTRY)
    return trainers
