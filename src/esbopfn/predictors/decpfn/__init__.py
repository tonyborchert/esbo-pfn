"""DecPFN: causal decoder predicting the total cost from a running trajectory prefix.

``model`` (network), ``train`` (training), ``predictor`` (inference). Predictor ids
``decpfn/<problem>/<head>/<size>[/<run>]`` and the same for ``decpfn-nostates``,
``decpfn-nodeadtime`` and ``decpfn-nodeadtime-nostates`` (factories
:func:`make_nostates_predictor`, :func:`make_nodeadtime_predictor` and
:func:`make_nodeadtime_nostates_predictor`). Variants starting with ``thesis`` are the imported
thesis checkpoints (:mod:`esbopfn.predictors.decpfn.legacy`).
Importing this package does not import torch.
"""

from __future__ import annotations

from typing import Any

from esbopfn.core import ProblemSpec


def _make(family: str, args: tuple[str, ...], spec: ProblemSpec, **kw: Any):
    from esbopfn.predictors.decpfn.predictor import load

    if len(args) not in (3, 4):
        raise ValueError(f"expected {family}/<problem>/<head>/<size>[/<run>], "
                         f"got {family}/{'/'.join(args)}")
    if args[0] != spec.name:
        raise ValueError(f"id names problem {args[0]!r}, predictor requested for {spec.name!r}")
    if args[1].startswith("thesis"):        # imported thesis checkpoint, old architecture
        from esbopfn.predictors.decpfn.legacy import load as load_legacy

        return load_legacy("/".join((family, *args)), spec, **kw)
    return load("/".join((family, *args)), spec, **kw)


def make_predictor(args: tuple[str, ...], spec: ProblemSpec, **kw: Any):
    """Registry factory for ``decpfn/<problem>/<head>/<size>[/<run>]``."""
    return _make("decpfn", args, spec, **kw)


def make_nostates_predictor(args: tuple[str, ...], spec: ProblemSpec, **kw: Any):
    """Registry factory for ``decpfn-nostates/...``."""
    return _make("decpfn-nostates", args, spec, **kw)


def make_nodeadtime_predictor(args: tuple[str, ...], spec: ProblemSpec, **kw: Any):
    """Registry factory for ``decpfn-nodeadtime/...``."""
    return _make("decpfn-nodeadtime", args, spec, **kw)


def make_nodeadtime_nostates_predictor(args: tuple[str, ...], spec: ProblemSpec, **kw: Any):
    """Registry factory for ``decpfn-nodeadtime-nostates/...``."""
    return _make("decpfn-nodeadtime-nostates", args, spec, **kw)
