"""``pfn train ftpfn``: thin wrapper around :func:`esbopfn.predictors.ftpfn.train.train`.

``run(request)`` is the :data:`esbopfn.training.TRAINERS` entry
(``"ftpfn": "esbopfn.training.train_ftpfn:run"``), :func:`register` adds it to a trainer table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from esbopfn.predictors.ftpfn.train import train

__all__ = ["register", "run", "train"]

ENTRY = "esbopfn.training.train_ftpfn:run"


def run(request: Any) -> Path:
    """Train FT-PFN for one :class:`~esbopfn.training.TrainRequest` (head and size are ignored,
    FT-PFN has one configuration)."""
    return train(request.problem, None, seed=request.seed, run=request.run,
                 steps=request.steps, time_limit=request.time_limit, device=request.device,
                 home=request.home, hparams_file=request.hparams)


def register(trainers: dict[str, Any] | None = None) -> dict[str, Any]:
    """Add ``"ftpfn"`` to ``trainers`` (default :data:`esbopfn.training.TRAINERS`)."""
    if trainers is None:
        from esbopfn.training import TRAINERS as trainers
    trainers.setdefault("ftpfn", ENTRY)
    return trainers
