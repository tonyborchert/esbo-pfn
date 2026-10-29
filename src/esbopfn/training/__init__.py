"""Training of the own models.

:mod:`.common` is the shared training core (:func:`~esbopfn.training.common.fit`),
:mod:`.registry` the model tree and model ids, :mod:`.heads` the crash-aware output heads,
:mod:`.normalization` the z-score statistics, :mod:`.single_trajectory_data` the LC-PFN/DecPFN
batches and :mod:`.hpo` the LC-PFN/DecPFN hyperparameter search.

``pfn train <model>`` looks the model up in :data:`TRAINERS`. Every entry is a callable taking a
:class:`TrainRequest` and returning the model directory. Entries are ``"module:attribute"``
strings resolved on first use so that importing this package stays light (no torch). A model
family adds itself by adding one line to :data:`TRAINERS`.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainRequest:
    """One ``pfn train`` invocation, resolved to a single (model, problem) pair.

    ``head`` and ``size`` are None for models without these axes (XDecPFN, FT-PFN). ``hparams``
    is an optional YAML file overriding ``configs/hparams/<model>.yaml``. ``steps`` overrides the
    number of optimizer steps, ``time_limit`` caps the training time in seconds.
    """

    model: str
    problem: str
    head: str | None = None
    size: str | None = None
    hparams: Path | None = None
    seed: int = 0
    run: str = "0"
    steps: int | None = None
    time_limit: float | None = None
    device: str = "auto"
    home: Path | None = None


TrainFn = Callable[[TrainRequest], Path]

#: ``pfn train`` model name -> trainer (callable or ``"module:attribute"``).
TRAINERS: dict[str, TrainFn | str] = {
    "lcpfn": "esbopfn.training.train_lcpfn:run",
    "lcpfn-last": "esbopfn.training.train_lcpfn:run",
    "decpfn": "esbopfn.training.train_decpfn:run",
    "decpfn-nostates": "esbopfn.training.train_decpfn:run",
    "decpfn-nodeadtime": "esbopfn.training.train_decpfn:run",
    "decpfn-nodeadtime-nostates": "esbopfn.training.train_decpfn:run",
    "xdecpfn": "esbopfn.training.train_xdecpfn:run",
    "ftpfn": "esbopfn.training.train_ftpfn:run",
}


def get_trainer(model: str) -> TrainFn:
    """Resolve the trainer of ``model`` from :data:`TRAINERS`."""
    try:
        entry = TRAINERS[model]
    except KeyError:
        raise KeyError(f"unknown model {model!r}, known: {', '.join(sorted(TRAINERS))}") from None
    if isinstance(entry, str):
        module, _, attr = entry.partition(":")
        entry = getattr(importlib.import_module(module), attr)
    return entry
