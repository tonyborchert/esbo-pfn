"""XDecPFN, the two-stage cross-trajectory predictor (thesis method chapter, XDecPFN).

Ids: ``xdecpfn/<problem>/default/default[/<run>]`` for a model trained with
``pfn train xdecpfn --problem P``, ``xdecpfn/<problem>/thesis/default`` for the imported thesis
checkpoint (:mod:`esbopfn.benchmarks.import_models`).
"""

from __future__ import annotations

from typing import Any

from esbopfn.core import ProblemSpec

MODEL = "xdecpfn"


def parse_args(model: str, args: tuple[str, ...], spec: ProblemSpec) -> str:
    """``(problem, variant, size[, run])`` (all optional) -> the full model id."""
    parts = list(args) + ["", "", "", ""][len(args):]
    problem = parts[0] or spec.name
    if problem != spec.name:
        raise ValueError(f"model id names problem {problem!r}, the predictor is built for {spec.name!r}")
    variant, size, run = parts[1] or "default", parts[2] or "default", parts[3] or "0"
    return f"{model}/{problem}/{variant}/{size}/{run}"


def make_predictor(args: tuple[str, ...], spec: ProblemSpec, **kw: Any):
    """Registry factory: load the model directory of the id, raise ``ModelNotTrained`` if absent.

    Keyword arguments: ``device`` (default ``cpu``), ``home`` (the ``PFN_HOME`` to look in).
    """
    from esbopfn.training.registry import ModelId, resolve

    model_id = parse_args(MODEL, args, spec)
    home = kw.pop("home", None)
    mid = ModelId.parse(model_id)
    if mid.variant == "thesis":
        path = resolve(model_id, home, hint="Import it with `pfn import thesis-models --from DIR`.")
        from esbopfn.predictors.xdecpfn.legacy import LegacyXDecPFNPredictor

        return LegacyXDecPFNPredictor.from_dir(path, spec, name=mid.short, **kw)
    path = resolve(model_id, home, hint=f"Train it with `pfn train {MODEL} --problem {spec.name}`.")
    from esbopfn.predictors.xdecpfn.predictor import XDecPFNPredictor

    return XDecPFNPredictor.from_dir(path, spec, name=mid.short, **kw)
