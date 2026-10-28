# Training

This page covers the training of the own single-trajectory models LC-PFN and DecPFN (and their
variants), the shared training core every own model uses, the model tree under
`$PFN_HOME/models/`, the hyperparameter files and the LC-PFN/DecPFN hyperparameter search.
XDecPFN and FT-PFN train on the same core with their own data loaders (see their modules).

## Quick start

```bash
pfn data gen --for train decpfn --problem cartpole       # traj/cartpole/1M and traj/cartpole/val
pfn train decpfn --problem cartpole                      # binned head, 1M
pfn train decpfn --problem cartpole --head gaussian --size 100k
pfn train lcpfn --problem boiler --head all --size all   # six models
pfn train all --problem all                              # every model, binned 1M where it applies
pfn models list
pfn models show decpfn/cartpole/binned/1M
```

`pfn train <model> --problem P [--head binned|gaussian|linear|all] [--size 100k|1M|all]
[--hparams FILE] [--seed S] [--run R] [--steps N] [--time-limit S] [--device auto|cuda|mps|cpu]`.
Models: `lcpfn`, `lcpfn-last`, `decpfn`, `decpfn-nostates`, `decpfn-nodeadtime` (Furuta only),
plus the models other packages register (`xdecpfn`, `ftpfn`). Head and size default to `binned`
and `1M`. `--steps` overrides the step count (debugging and smoke runs, a count beyond one pass
over the data starts a fresh seeded pass), `--time-limit` caps the training time. A finished model
directory is never retrained, an unfinished one resumes from its checkpoint. On a cluster, wrap the
command in `pfn submit --kind train -- pfn train ...` (see `docs/CLUSTER.md`).

## Model ids and the model tree

A model id is `<model>/<problem>/<variant>/<size>[/<run>]`, the run defaults to `0`:

| Id | Model |
|---|---|
| `lcpfn/<p>/<head>/<size>` | LC-PFN |
| `lcpfn-last/<p>/<head>/<size>` | LC-PFN trained with the query at the final grid point only |
| `decpfn/<p>/<head>/<size>` | DecPFN |
| `decpfn-nostates/<p>/<head>/<size>` | DecPFN-no-St, tokens `(J, ΔJ, t)` only |
| `decpfn-nodeadtime/furuta/<head>/<size>` | DecPFN on the Furuta prior without dead time (hardware ablation) |
| `xdecpfn/<p>/default/default`, `ftpfn/<p>/default/default` | cross-trajectory models |

The directory of an id is `$PFN_HOME/models/<model>/<problem>/<variant>/<size>/<run>/` with

- `model.pt`: state dict and model configuration (written last, its presence marks a trained model),
- `config.yaml`: resolved hyperparameters, dataset ids, seed, step count, git hash, provenance,
- `calib.json`: normalization statistics and bar edges (layout below),
- `metrics.json`: loss curve, learning rate, gradient norms, validation results, timing.

The same ids are predictor ids: `predictors.get_predictor("decpfn/cartpole/binned/1M", "cartpole")`
returns a `Predictor`, and an absent model raises `ModelNotTrained` whose message names the
`pfn train` command. `esbopfn.training.registry` resolves ids (`model_dir`, `resolve`,
`load_model`, `list_models`).

## The thesis recipe

This is what the code does (thesis method chapter and the training appendix).

**Data.** The single-trajectory prior pools `traj/<p>/1M` from `pfn data gen`. The 100k models
use the first 100,000 rows, the 1M models all 1,000,000 rows, each row once in a seeded random
order: `floor(N / 128)` optimizer steps (781 and 7,812). Validation uses separate seed streams:
3,000 rows for DecPFN (`traj/<p>/val`), the first 1,024 of them for LC-PFN (`traj/<p>/val_lcpfn`).

**Grids.** Both models place `N_grid` points per trajectory (`esbopfn.grids`). DecPFN uses the
exact-stride decoder grid `t_l = l · stride`, `l = 1 .. N_grid − 1`, and predicts the horizon
`T`. LC-PFN uses `t_l = floor(l T / N_grid)`, `l = 1 .. N_grid`, the last point being `T`.

**DecPFN tokens and loss.** A token is `(J_{t_l}, ΔJ_l, ξ_{t_l}, Σ ξ̇, t_l)` with the covariates of
`grids.tokens` (levels at the grid point, rates summed over the interval at the raw step
resolution, increments of the levels when a problem has no measured rates). The causal pre-LN
decoder applies the head at every position against the target `J_T` (`+inf` on a crash). Only
positions strictly before the crash are scored, and the batch loss is the mean over all scored
positions.

**LC-PFN tokens and loss.** Context tokens `(t_l, J_{t_l})`, query tokens `t_l`. One cutoff
`κ ~ U{1 .. N_grid − 1}` per batch, the first `κ` points are the context and every remaining point
is a query (`lcpfn-last`: only the final point). Rows that crash at or before `t_κ` are removed
from that batch and not requeued. A query at or after the crash has the crash atom as target. The
loss is the mean over all queries of the batch.

**Architectures.** DecPFN: `Linear(F + 1 → d)` on the z-scored inputs and time, pre-LN
Transformer blocks under a causal mask, a final layer norm and a single linear output layer.
LC-PFN: the released LC-PFN architecture, linear encoders of `t` and `J` summed for context tokens,
post-LN blocks under the PFN mask (context attends to context, a query to the context and itself),
attention output projection and second FFN layer zero-initialized, and the two-layer FFN head
`Linear(d, 2d) − GELU − Linear(2d, outputs)`. Both: `d_ff = 2 d_model`, GELU, `n_heads = d_model / 16`,
dropout as configured, layer norms with `γ = 1, β = 0`.

**Heads.** `binned`, `linear`: crash-aware bar distribution with `N_bin = 1000` bins plus the crash
atom (`training/heads.py`). The loss is `−log w_∞` on a crash, else
`−log(1 − w_∞) − log w_k + log(b_k − b_{k−1})`. `binned` places the edges at equal-mass quantiles of
prior samples, `linear` at equal widths over their range. LC-PFN samples the full trajectory value
range (every grid point), DecPFN the horizon `J_T`. Targets outside the edges are clamped into the
outermost bins. `gaussian`: outputs `μ, log σ², logit w_∞`, the loss is `−log w_∞` on a crash,
else `−log(1 − w_∞) + ½ log(2πσ²) + (y − μ)²/(2σ²)`.

**Normalization.** Every input channel is z-scored with mean and standard deviation of a seeded
subset of the training pool (20,000 rows, bar edges of DecPFN from up to 100,000 horizon values).
The time is z-scored over its known range, `{1 .. T}` for LC-PFN and `{1 .. T − 1}` for DecPFN.
The targets of all heads live in the space of the z-scored `J` channel, at inference the
predictions are mapped back to raw cost units.

**Optimization** (`training/common.py`). AdamW, linear warmup from 0 over the first 25 % of the
steps, then cosine decay to 0, stepped per optimizer step, gradient norm clipped to 1, batch 128,
bf16 autocast on CUDA. Seeds: the parameter initialization draws from `(seed, "init")`, dropout
from `(seed, "train")`, the data order from `(seed, "shuffle", pass)` and the LC-PFN cutoffs from
`(seed, "cutoff")`, all through `esbopfn.util.seeds`.

**Validation.** The validation NLL of the horizon prediction at the grid cutoffs between 20 % and
99 % of the `N_grid − 1` possible cutoffs (`κ = round(0.2 L) + 1 .. round(0.99 L)`), in raw cost
units (the log standard deviation of the `J` normalization is added to every finite target),
averaged over every (row, cutoff) pair whose row has not crashed by the cutoff. `metrics.json`
reports it as `val_nll`, next to `val_nll_0_99`.

## Hyperparameter files

`src/esbopfn/configs/hparams/lcpfn.yaml` and `decpfn.yaml` hold the thesis configuration table,
one row per (size, head), plus the fixed block and the search definition:

```yaml
fixed: {batch_size: 128, d_k: 16, d_ff_factor: 2, n_bin: 1000, warmup_frac: 0.25,
        clip_grad_norm: 1.0, bf16: true}
configs:
  - {size: 1M, head: binned, n_layers: 6, d_model: 384, lr: 5.6e-4, weight_decay: 1.0e-4, dropout: 0.0}
  ...
hpo: {problem, start, stage1, stage2, selection}
```

`lcpfn-last` reads `lcpfn.yaml`, `decpfn-nostates` and `decpfn-nodeadtime` read `decpfn.yaml`
(`inherited_by`). The selected values are those of the thesis on every problem.

## Hyperparameter search

```bash
pfn hpo decpfn --problem cartpole --size 100k                 # both stages
pfn hpo lcpfn --problem cartpole --size 1M --stage 1
pfn submit --kind train --array 0-19 -- pfn hpo decpfn --size 1M --stage 2   # 20 shards
```

Stage 1 sweeps, from the start configuration (`d_model = 512`, `N_layer = 3`, `linear` head,
weight decay `1e-6`, dropout 0), first the learning rate over `{1.0, 1.8, 3.2, 5.6, 10} × 1e-4`,
then the weight decay over `{1e-6, 1e-4, 1e-2, 1e-1}`, then the dropout over
`{0, 0.1, 0.2, 0.35, 0.5}`, each step keeping the best values of the steps before. Stage 2 trains,
for each head, the grid `d_model ∈ {256, 384, 512, 1024} × N_layer ∈ {1, 2, 3, 6, 12}` with the
stage-1 values, each run capped at 25 s (100k) or 250 s (1M) of training time. A capped run
follows the full learning-rate schedule and is cut off when the cap is reached. Every head takes
the cell with the lowest validation NLL of its own grid.

Outputs under `$PFN_HOME/hpo/<model>/<problem>/<size>/`:

- `stage1/<step>_<param>_<value>/`, `stage2/<head>/d<d_model>_l<N_layer>/`: trained model
  directories with `result.json`,
- `trials.csv`: every finished trial,
- `pareto.json`: per head the Pareto front over (training time, validation NLL) and the best cell,
- `hparams_candidate.yaml`: the selected configuration in the hyperparameter-file format,
- `gridsearch_panel.pdf`: training wall-clock against validation NLL, marker size proportional to
  the parameter count, one marker per head.

Every trial is idempotent and resumable. `--shard i/n` runs every n-th trial of the current step,
so stage 1 needs one round of shards per step, and stage 2 shards over its 60 cells. The summary
files are rewritten at the end of every invocation.

To train with the result, pass the candidate file:

```bash
pfn train decpfn --problem cartpole --size 100k --head binned \
    --hparams $PFN_HOME/hpo/decpfn/cartpole/100k/hparams_candidate.yaml
```

## Training core API (for model packages)

```python
from esbopfn.training import TRAINERS, TrainRequest
from esbopfn.training.common import TrainConfig, fit, build_model, train_config_from_row
from esbopfn.training import registry
```

- `fit(model, batches, loss_fn, cfg, out_dir, evaluate=None) -> FitResult`. `batches` is an
  iterator or, better, a callable `start_step -> iterator`, so a resumed run continues the data
  stream exactly. Batches (tensors in dicts, lists, tuples or dataclasses) are moved to the device
  before `loss_fn(model, batch)`. `out_dir/ckpt_last.pt` is written every `checkpoint_every`
  steps, `metrics.json` at the end.
- `build_model(factory, seed)` constructs a model with seeded initialization without touching the
  global RNG.
- `registry.save_model(out_dir, state_dict, model_cfg, calib, train_cfg, metrics)`,
  `registry.load_model(path)`, `registry.resolve(model_id)`, `registry.model_dir(model_id)`.
- `TRAINERS` maps a `pfn train` model name to `run(TrainRequest) -> Path` (a callable or a
  `"module:attribute"` string). A model package registers itself by adding its line there.

## `calib.json`

```json
{
  "version": 1, "model": "decpfn", "problem": "cartpole", "head": "binned",
  "features": ["J", "dJ", "x_cart", "phi_pole", "sum_xdot_cart", "sum_phidot_pole", "t"],
  "input": {"mean": [...], "std": [...]},
  "time": {"lo": 1, "hi": 999, "mean": 500.0, "std": 288.4},
  "target": {"mean": 12.3, "std": 45.6},
  "edges": [...],
  "n_bin": 1000, "grid": [15, 30, ...], "T": 1000,
  "samples": {"rows": 20000, "edge_rows": 100000, "seed": 0, "pool": "..."}
}
```

`input` covers every feature with the time last. `target` holds the statistics of the `J`
channel, `edges` the `N_bin + 1` bar edges in that normalized space (`null` for `gaussian`). Raw
edges are `edges · target.std + target.mean`.
