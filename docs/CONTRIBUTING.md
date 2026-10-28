# Contributing

## Setup

```
uv sync
uv run pytest -q
```

Unit tests run in seconds on a laptop. Tests marked `cpp`, `gpu`, `slurm` or `hardware` run when
the resource is found or when `--run-<marker>` is passed.

## Conventions

- Python 3.12, one package `esbopfn` under `src/`, console script `pfn`.
- Type hints everywhere, dataclasses for configs and records, `logging` instead of `print` (the
  command modules print their results to stdout).
- No module-level side effects, no configuration read at import time, no `sys.path` edits.
  Configuration reaches code through explicit arguments. The only environment variables are
  `PFN_HOME` and `PFN_PROFILE`.
- Every random draw takes an explicit `numpy.random.Generator` or torch generator derived with
  `esbopfn.util.seeds.substream(root_seed, *labels)`. No global RNG.
- Small modules. Each own predictor has `model.py`, `data.py`, `train.py` and `predictor.py`.
- Steps are 1-based as in the thesis.
- Docs and docstrings in American English, without semicolons, and "early stopping" without a
  hyphen.

## Extension points

Each of these is a new file, the registry code stays unchanged.

| what | where | exposes |
|---|---|---|
| CLI command | `src/esbopfn/commands/<name>.py` | `register(subparsers)` |
| benchmark kind | `src/esbopfn/benchmarks/kinds/<kind>.py` | `KIND = BenchmarkKind(...)` |
| plot group | `src/esbopfn/plots/<group>.py` (+ `configs/benchmarks/<group>.yaml`) | `GROUP = PlotGroup(...)` |
| cluster | `src/esbopfn/configs/cluster/<name>.yaml` | see `docs/CLUSTER.md` |

## Tests

Add tests under `tests/unit/` for pure Python code and under `tests/<marker>/` with the matching
marker for code that needs a resource.
