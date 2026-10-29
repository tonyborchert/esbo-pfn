# Contributing

## Setup

```
uv sync
uv run pytest -q
uvx ruff check src tests tools
```

`pyproject.toml` holds the ruff standard: line length 100, target Python 3.12, rule sets `E F I UP
B`. Format new or rewritten code with `uvx ruff format <file>` or `--range`. The existing modules
are hand-wrapped, so a tree-wide `ruff format` is not applied, and `E501` tolerates existing lines
up to 120 columns.

Unit tests run in seconds on a laptop. Tests marked `cpp`, `gpu`, `slurm` or `hardware` run when
the resource is found or when `--run-<marker>` is passed.

## Conventions

- Python 3.12, one package `esbopfn` under `src/`, console script `pfn`.
- Type hints everywhere, dataclasses for configs and records, `logging` instead of `print` (the
  command modules print their results to stdout).
- No module-level side effects, no configuration read at import time, no `sys.path` edits.
  Configuration reaches code through explicit arguments. The only environment variables read are
  `PFN_HOME` and `PFN_PROFILE`, plus `PFN_SLURM_ACCOUNT_CPU`/`PFN_SLURM_ACCOUNT_GPU` and the
  `${HPCWORK}`/`${PROJECT}` placeholders of a cluster profile (see [CLUSTER.md](CLUSTER.md)).
- Every random draw takes an explicit `numpy.random.Generator` or torch generator derived with
  `esbopfn.util.seeds.substream(root_seed, *labels)`. No global RNG.
- Small modules. Each own predictor has `model.py`, `train.py` and `predictor.py`, plus `data.py`
  where its data loader is its own (XDecPFN, FT-PFN). LC-PFN and DecPFN share
  `training/single_trajectory_data.py`.
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
| cluster | `src/esbopfn/configs/cluster/<name>.yaml` | see [CLUSTER.md](CLUSTER.md) |

## Tests

Add tests under `tests/unit/` for pure Python code and under `tests/<marker>/` with the matching
marker for code that needs a resource.
