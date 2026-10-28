"""FT-PFN adapted to controller trajectories (thesis method chapter, adapting FT-PFN).

Ids: ``ftpfn/<problem>/default/default[/<run>]`` for a model trained with
``pfn train ftpfn --problem P``, ``ftpfn/<problem>/thesis/default`` for the imported thesis
checkpoint (:mod:`esbopfn.benchmarks.import_models`).
"""

from __future__ import annotations

from typing import Any

from esbopfn.core import ProblemSpec

MODEL = "ftpfn"


def make_predictor(args: tuple[str, ...], spec: ProblemSpec, **kw: Any):
    """Registry factory: load the model directory of the id, raise ``ModelNotTrained`` if absent.

    Keyword arguments: ``device`` (default ``cpu``), ``home`` (the ``PFN_HOME`` to look in).
    """
    from esbopfn.predictors.xdecpfn import parse_args
    from esbopfn.training.registry import ModelId, resolve

    model_id = parse_args(MODEL, args, spec)
    home = kw.pop("home", None)
    mid = ModelId.parse(model_id)
    if mid.variant == "thesis":
        path = resolve(model_id, home, hint="Import it with `pfn models import-thesis --from DIR`.")
        from esbopfn.predictors.ftpfn.legacy import LegacyFTPFNPredictor

        return LegacyFTPFNPredictor.from_dir(path, spec, name=mid.short, **kw)
    path = resolve(model_id, home, hint=f"Train it with `pfn train {MODEL} --problem {spec.name}`.")
    from esbopfn.predictors.ftpfn.predictor import FTPFNPredictor

    return FTPFNPredictor.from_dir(path, spec, name=mid.short, **kw)
