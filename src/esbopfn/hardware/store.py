"""Reading rig runs (``hardware_optimization`` units) back from the benchmark database."""

from __future__ import annotations

from collections.abc import Sequence

from esbopfn.benchmarks.db import Db
from esbopfn.hardware import records
from esbopfn.hardware.records import HwRun

KIND = "hardware_optimization"


def load_runs(rig: str, db: Db | None = None, provenance: str | None = None,
              methods: Sequence[str] | None = None, include_unfinished: bool = False
              ) -> list[HwRun]:
    """Runs of ``rig`` sorted by (method, seed). ``provenance`` filters ``native``/``thesis``,
    ``include_unfinished`` adds runs that are still pending (their trials so far)."""
    db = db or Db()
    filters = {"rig": rig}
    if provenance is not None:
        filters["provenance"] = provenance
    units = db.query(KIND, status=None if include_unfinished else "done", **filters)
    runs = []
    for u in units:
        if methods is not None and u.key["method"] not in methods:
            continue
        blob = u.blob_dir or db.blob_dir(KIND, u.key)
        trials = records.read_trials(blob)
        runs.append(HwRun(rig=rig, method=u.key["method"], seed=int(u.key["seed"]),
                          trials=trials, key=u.key))
    runs.sort(key=lambda r: (r.method, r.seed))
    return runs
