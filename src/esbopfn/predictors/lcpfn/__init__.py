"""LC-PFN: the learning-curve PFN adapted to cost trajectories with a crash-aware head.

``model`` (network), ``train`` (training), ``predictor`` (inference). Predictor ids
``lcpfn/<problem>/<head>/<size>[/<run>]`` and ``lcpfn-last/...`` (factory
:func:`make_last_predictor`). Variants starting with ``thesis`` are the imported thesis
checkpoints (:mod:`esbopfn.predictors.lcpfn.legacy`). Importing this package does not import torch.
"""

from __future__ import annotations

from typing import Any

from esbopfn.core import ProblemSpec


def _make(family: str, args: tuple[str, ...], spec: ProblemSpec, **kw: Any):
    from esbopfn.predictors.lcpfn.predictor import load

    if len(args) not in (3, 4):
        raise ValueError(f"expected {family}/<problem>/<head>/<size>[/<run>], "
                         f"got {family}/{'/'.join(args)}")
    if args[0] != spec.name:
        raise ValueError(f"id names problem {args[0]!r}, predictor requested for {spec.name!r}")
    if args[1].startswith("thesis"):        # imported thesis checkpoint, old architecture
        from esbopfn.predictors.lcpfn.legacy import load as load_legacy

        return load_legacy("/".join((family, *args)), spec, **kw)
    return load("/".join((family, *args)), spec, **kw)


def make_predictor(args: tuple[str, ...], spec: ProblemSpec, **kw: Any):
    """Registry factory for ``lcpfn/<problem>/<head>/<size>[/<run>]``."""
    return _make("lcpfn", args, spec, **kw)


def make_last_predictor(args: tuple[str, ...], spec: ProblemSpec, **kw: Any):
    """Registry factory for ``lcpfn-last/...``."""
    return _make("lcpfn-last", args, spec, **kw)
