# Simulation benchmarks

Code: `src/esbopfn/benchmarks/` (`db.py`, `instances.py`, `tasks.py`, `metrics.py`, `groups.py`,
`import_thesis.py`, `import_timing.py`, `kinds/*.py`), specs in `src/esbopfn/configs/benchmarks/`.
Thesis references: `S` = setup.tex, `SR` = sim_results.tex, `A` = appendix.tex (line numbers
approximate). The hardware kinds are documented with the hardware stack.

## Overview

Every benchmark result is a *unit* of a *kind*, identified by a JSON key and stored once in the
database `$PFN_HOME/benchmarks/db.sqlite` with its files under
`$PFN_HOME/benchmarks/blobs/<kind>/<key_hash>/`. A plot group declares the units it needs in a
spec `configs/benchmarks/<group>.yaml`. `pfn bench plan|run <group>` expands the spec and computes
what is missing. A unit is never computed twice.

The kinds depend on each other in this order (`groups.KIND_ORDER`):

```
instances -> fields -> (task tables) -> prediction, prediction_ladder, prediction_timing
instances -> optimization, optimization_timing, landscapes
generation_speed (needs the C++ binaries)
thesis_series (written only by the importer)
```

Seeds: every random draw derives from `BENCH_SEED = 20261029` (`benchmarks/instances.py`) through
`substream`. Equal (problem, instance, seed) give equal initial designs across strategies.

## Kinds

Key fields are exact, blob files are relative to the unit's blob directory.

### `instances` (S:530-532, S:632-634)

Instance `i` of a problem is drawn from the universal prior with
`substream(BENCH_SEED, "instances", problem, i, attempt)` for `attempt = 0, 1, ...` and kept at the
first attempt that passes the degeneracy screen (100 controllers, 10 repeats, at least 25 passing).
It depends only on `(seed, problem, i)`, so the list is append-only. Optimization uses instances
0-19, the prediction fields all 50.

- Key `{problem, seed, count}`, thesis import `{problem, provenance: "thesis"}`.
- Blob `instances.json`: `{problem, root_seed, count, screen{n_theta, repeats, min_pass},
  instances: [{index, attempt, attempts, phi{}, phi_hat{}, noise_std{}, extras{}, seed,
  screen{pass_count, theta_screened, trials_used}, rejected_trials, seconds}]}`.
- `instances.load_instances(problem, root_seed)` returns `list[Instance]`.

### `fields` (S:525-536)

ESBO-Oracle runs whose trials become prediction tasks. `oracle65`: 65 trials, 50 instances, seeds
per problem boiler 4, threetank_pi 4, pt2_deadtime 7, cartpole 8, furuta 10. `oracle181`:
`N_max + 1` trials (181 on Cartpole and Furuta, 136 on PT2), 50 instances x 2 seeds, for the
context ladder. The oracle runs through `run_optimization` with `n_max = n_trials` and
`t_budget = n_trials * T`, so it always reaches `n_trials` trials and the time budget never cuts
the run. The loop simulates every trial in full, the full uncut trajectories are kept.

- Key `{problem, field, instance, seed, n_trials, instances_seed}`.
- Blob `trials.npz`: `theta (n, d) float64`, `t_obs (n,) int32` (observed length),
  `crash_step (n,) int32` (1-based, 0 = no crash when run to the end), `success (n,) bool`,
  `J (n, T) float32` (uncut, +inf from the crash step), `x (n, T, C) float32`,
  `xdot (n, T, C_rate) float32`. `field.json`: `{problem, field, instance, seed, run_seed,
  instance_params, n_trials, T, d, channels, rate_channels}`. `run.json`: the oracle's
  `RunRecord` (no curves).
- `fields.read_field(blob_dir)` returns a `FieldRun` with `trial(i, t)`, `context(n)` (the first
  `n` trials cut at their observed length) and `target(i)`.

### Task tables (`benchmarks/tasks.py`, S:525-536)

Not a kind but frozen files `$PFN_HOME/benchmarks/tasks/<problem>/<name>.parquet` plus a `.json`
sidecar with a digest (sha256, first 16 hex digits, of `task_id, crash_step, crashed, target_J,
selected, crash_selected`). They are built the first time a plan finds the whole field done.

- `tasks-oracle65`: query = trial `n + 1` of a run, context = trials `1..n`, for
  `n in (0, 2, 4, 8, 16, 32, 64)`. A query is crashed if its trajectory crashes when run to the
  end. Per `(problem, n)` the first 50 non-crashed queries in seed-major run order (seed 0 of every
  instance, then seed 1, ...) are `selected`, and on problems that can crash the first 50 crashed
  ones `crash_selected`.
- `ladder-oracle181`: query = the last trial of the first 50 seed-major runs whose last trial does
  not crash, context = the first `n` trials for the rungs PT2 `0,1,2,4,8,16,33,67,134` and
  Cartpole/Furuta `0,1,2,5,11,22,45,90,180`.
- Columns: `task_id, problem, field, run_id, instance, seed, run_order, n_ctx, query_index, T,
  t_obs, crash_step, crashed, target_J, selected, crash_selected`.
- `tasks.verify_tasks(problem)` rebuilds a table from its field and compares digests.

### `prediction` (S:542-576)

One predictor on one problem's frozen context tasks. The query is cut at the five cutoffs
10/30/50/70/90 % of `T`, each snapped to the last predictable step at or before `f * T` (the first
predictable step when none lies before it, flagged `cutoff_fallback`). Every prediction is scored
with the conditioned log-likelihood (`predictors.base.conditioned_log_likelihood`: conditioned on
`J >= J_t`, crash mass kept, floor 1e-10). Crash-aware predictors are also evaluated at every
predictable step before the end of the trial (before the crash step for crashed queries), the data
of the stop curves. Context-blind predictors get no context. Scored tasks: `selected` at the
requested `n`, plus `crash_selected` for crash-aware predictors.

- Key `{problem, model, checkpoint, tasks, tasks_digest, n, part}`. `checkpoint` is the sha256
  (16 hex digits) of the weights (`none` without weights, `missing` if not trained yet). `part =
  "i/k"` takes every k-th task starting at i.
- Blob `predictions.parquet`, one row per scored cutoff: `task_id, run_id, instance, seed, n_ctx,
  query_index, crashed, crash_step, target_J, cutoff_frac` (NaN on stop-grid rows),
  `cutoff_fallback, cutoff_step, J_obs, ll, p_crash, dist` (`bar` | `gaussian`), `edges, probs`
  (float32 lists, only on the five cutoff rows of bar predictions), `mean, var, seconds`.
  `meta.json`: `{key, model, context_aware, crash_aware, n_rows, n_tasks, seconds, device, host,
  cpu, torch, torch_threads}`.

### `prediction_ladder` (SR:156-189)

The same scoring on the ladder table, no stop-grid rows. Key `{problem, model, checkpoint, tasks,
tasks_digest, n, part}`, blob as `prediction`.

### `prediction_timing` (S:560-563)

Median inference seconds per prediction: the first `queries` selected tasks at `n_ctx`, predicted
at the cutoff `cutoff_frac`, `warmup` untimed calls then `repeats` timed calls per query (CUDA
synchronized). Run it on the machine whose times are wanted, the device is recorded.

- Key `{problem, model, checkpoint, tasks, tasks_digest, n_ctx, queries, repeats, warmup,
  cutoff_frac}`, thesis import `{problem, model, n_ctx, provenance, source_model}`.
- Blob `timing.json`: `{seconds_median, seconds_mean, seconds_p90, n_calls, device, host, cpu,
  torch, torch_threads, measured_at}`.

### `optimization` (S:630-681)

One run per (strategy, instance, seed) under the thesis caps `N_max = 3*15*min(d,4)` trials and
`T_budget = 15*T*min(d,4)` steps (S eq:setup:budgets), whichever binds first.

- Key `{problem, instance, strategy, params, seed, budget: "capped", instances_seed}`. `params` =
  registry defaults merged with the spec's overrides, model ids resolved, plus `checkpoint` when a
  model is used. Optional `n_max`, `t_budget` override the caps (tests only). Thesis import adds
  `provenance, cell, label` and uses `instances_seed: "thesis"`.
- Blob `run.json`: `RunRecord.to_json()`: `{problem, strategy, params, seed, instance_seed, T,
  n_init, n_max, t_budget, trials[{theta, t_obs, J_obs, crashed, success, complete, J_true, source,
  propose_s, stop_rule_s, end_s, J}], incumbent_index[], incumbent_J[], timing{gp_acq, segment_gp,
  tabpfn_forecast, model_forecast, model_acq}, wall_s, version}` (curves off, the incumbent trace
  is always recorded).

### `optimization_timing` (S:671-681)

As `optimization` on instances 0-2 with `budget: "timing"`, run in one process with
`torch.set_num_threads(1)`. Blob `run.json` and `environment.json` `{device, host, cpu, torch,
torch_threads}`. Submit these units one per core.

### `generation_speed` (A:879-909)

Wraps `generators.speed.measure` (the binaries' `speed` subcommand, one core). Key `{problem,
count, sets, budget_s, seed}`. Blob `speed.json`: `{problem, traj_s_per_10k, traj_acceptance,
system_acceptance, ftpfn_s_per_10k, xdecpfn_s_per_10k, single_trajectories, xdecpfn_sets,
ftpfn_sets, system_candidates, host, cpu}`.

### `landscapes` (A:860-865, S:382-394)

- `unit: "slice"`: key `{unit, problem, instance, slice_seed, grid, repeats, instances_seed}`. A
  slice through the center of Theta: the whole interval on 80 points for `d = 1`, the plane for
  `d = 2`, a random oblique plane (two orthonormal Gaussian directions, the widest of up to 200
  draws) on a 24 x 24 grid for `d > 2`, extent as large as fits inside Theta, 6 repeats per point.
  Blob `landscape.npz`: `coords, thetas, p_crash, cost_mean` (mean `J_T` of successful repeats),
  `cost_mean_nocrash, center, u, v, extent`. The figure colors `metrics.cost_rank(cost_mean)` and
  `p_crash`.
- `unit: "prior_samples"`: key `{unit, problem, instance, n, max_unsuccessful, instances_seed}`.
  10 controllers `theta ~ U(Theta)` on screened instance 0, an unsuccessful trajectory is kept only
  while fewer than 5 kept ones are unsuccessful. Blob `samples.npz`: `theta, J, success,
  crash_step`.

### `thesis_series`

Aggregated thesis series, written only by the importer. Key `{figure, problem, series, variant,
provenance: "thesis"}` (`problem = "pooled"` for cross-problem series). Blob `series.json`:
`{figure, problem, series, variant, source, columns[], data{column: [values]}, meta{}}`.

## Metrics (`benchmarks/metrics.py`)

Pure functions over data frames and run records.

Prediction (S:542-576, subsubsec:setup:prediction-metrics):

| function | definition |
|---|---|
| `trial_scores(rows)` | per non-crashed query the mean of its cutoff LLs (five cutoffs) |
| `collapse_context_blind(scores)` | DecPFN, DecPFN-noStates, LC-PFN pooled over `n` (`n_ctx = -1`) |
| `median_scores(scores)` | median per (problem, model, n) |
| `med_d(scores, participants)` | per problem the median of every score of the declared participants |
| `ll_norm(scores, med)` | `score - med_D` of the problem |
| `pooled_median(normed)` | median of `ll_norm` pooled over problems per (series, n) |
| `pareto_front(points)` | points not dominated in (median inference time, score) |
| `stop_curve(rows, tasks)` | stop at the first step with `p_crash >= tau`, `tau` over every emitted probability: mean steps saved on crashed queries (`crash_step - stop_step`, 0 if never stopped) vs false-stop share on non-crashed queries |
| `steps_saved_at(curve, 0.1)` | largest mean steps saved at a false-stop share of at most 0.1 |
| `ladder_curve(rows)` | mean over cutoffs per query, then mean over queries |

The stop curves use the 50 crashed and 50 non-crashed queries of each `(problem, n)` (context-aware
predictors, one curve per `n`) or all of them pooled over `n` (context-blind predictors).

Optimization (S:630-681, subsubsec:setup:optimization-metrics):

| function | definition |
|---|---|
| `incumbent_vs_trials(rec, 15*min(d,4))` | incumbent after 1..n trials |
| `incumbent_vs_time(rec, time_grid(T_budget, 300))` | incumbent of the last finished trial at each simulated time |
| `j_best(finals, strategies)` | best finite final incumbent of the declared strategies per instance |
| `j_esrs(finals)` | ESRS's final incumbent per instance |
| `scaled_regret(J, J_best, J_ESRS)` | `(J - J_best) / (J_ESRS - J_best)`, `inf` without incumbent, NaN if `J_ESRS - J_best <= 1e-9` |
| `problem_median`, `cross_problem_mean` | median over instances per problem, mean over problems |
| `log10_display(x, 1e-4)` | log10 with display floor 1e-4, taken last |
| `regret_at_fraction(rec, ..., 0.5 or 1.0)` | per-run scaled regret at 50 / 100 % of `T_budget` (violins) |
| `mean_crash_count(records, grid)` | mean over runs of the number of crashed trials finished by each time |
| `timing_split(records)` | decision time per segment (GP acquisition, Transformer acquisition, Segment-GP forecast, Transformer forecast), summed per run and averaged over runs |

## Plot-group specs (`benchmarks/groups.py`)

`configs/benchmarks/<group>.yaml` holds `kinds`, shared `problems` and `instances: {count, seed}`,
and one section per kind. `groups.load(name)` reads a spec and `groups.units(cfg)` returns its
`(kind, key)` pairs in dependency order. Prediction sections list `models` (strings or `{id, n,
problems, parts}`, `{p}` is replaced by the problem) and `n`. Optimization sections list
`strategies` (`{id, params, label}`), `instances` and `seeds`. `participants` declares the model set
of `med_D` and `reference_strategies` the set of `J_best`, so a figure's normalization does not
depend on what else is in the database.

| spec | figures | units |
|---|---|---|
| `priors` | fig:setup:prior-samples, fig:app:prior-landscapes | instances, landscapes (instances 0-2, slice seed 1, prior samples) |
| `single_trajectory` | fig:sim:lcpfn-decpfn-heads, fig:sim:lcpfn-decpfn-stop-sheet | DecPFN, DecPFN-noStates, LC-PFN x {binned, gaussian, linear} x {100k, 1M} on `oracle65`, timing at `n = 0` |
| `tabpfn` | fig:sim:tabpfn-panel | Trajectory-TabPFN L = 25/50/100/250 (n >= 2), Segment-TabPFN rate/split (n >= 4), plus the `prediction_summary` participants |
| `cross_trajectory` | fig:sim:context-sweep, fig:sim:ftpfn-xdecpfn-stop-sheet | XDecPFN, FT-PFN on `oracle65` and the ladder (`oracle181`, PT2/Cartpole/Furuta) |
| `prediction_summary` | fig:sim:fig3-cross, fig:app:fig3-sheet, fig:sim:fig2-sheet | DecPFN, LC-PFN, DecPFN-noStates (binned 1M), XDecPFN, FT-PFN, TabPFN L = 50, Segment-TabPFN split, Segment-GP, which are also the `participants` |
| `optimization` | fig:sim:optplay-cross, -violin-cross-full, -crashes, -timing and the appendix sheets and variant violins | the nine thesis strategies, the six DecPFN/DecPFN-H threshold cells, the six further pure-loop cells, 5 problems x instances 0-19, timing with seven strategies on instances 0-2 |
| `generation_speed` | tab:app:prior-cost | one unit per problem |

The live-inference figures (fig:sim:lcpfn-decpfn-example, fig:sim:ftpfn-xdecpfn-example,
fig:sim:landscape-furuta) and fig:app:hpo-gridsearch need trained models or HPO results, not units.

## Running

```
pfn bench plan prediction_summary            # missing units per kind
pfn bench run priors                         # everything, in dependency order
pfn bench run prediction_summary --kind fields --shard 3/40
```

`pfn bench run` re-plans after each pass, so units whose inputs appear during the run (prediction
units after the field is complete) are computed in the same call. `--kind` restricts a plan to one
dependency stage. Large prediction units can be split with `parts: k` on a model entry.

### On Slurm

Run the stages as separate array jobs chained with `--after`:

```
pfn submit --kind gen --name inst -- pfn bench run prediction_summary --kind instances
pfn submit --kind gen --array 0-199 --after <inst> --name field -- \
    pfn bench run prediction_summary --kind fields
pfn submit --kind bench_gpu --array 0-39 --after <field> --name pred -- \
    pfn bench run prediction_summary --kind prediction
pfn submit --kind bench --array 0-299 --cpus 1 --name opt -- \
    pfn bench run optimization --kind optimization
```

`--array a-b` appends `--shard i/n` to the command. Every shard plans the same sorted unit list
and takes every n-th unit, so shards never collide. Timing units (`prediction_timing` on the
workstation, `optimization_timing` with one CPU per task) should run where their times are meant.

## Importing the thesis results

```
pfn import thesis-results --from /path/to/bachelorarbeit [--raw DIR] [--overwrite]
pfn import thesis-models --from DIR                      # XDecPFN and FT-PFN checkpoints
```

`import_thesis_results(root)` reads every result file below the `bachelorarbeit` root that exists,
`--raw` directories are searched for raw optplay trees `<problem>@g0.05@bcapped[@timing]/
phi_NNNN/runs/<cell>.json`. Units carry `provenance: "thesis"` and the thesis names are mapped:

| thesis | id |
|---|---|
| `decpfn/epochs{10,100}/head<h>[/states0]` | `decpfn[-nostates]/<p>/<h>/{100k,1M}` |
| `lcpfn_zscore/epochs{10,100}/head<h>` | `lcpfn/<p>/<h>/{100k,1M}` |
| `lcpfn/epochs{10,100}/head<h>` (before the z-scored inputs) | `lcpfn/<p>/<h>/{100k,1M}/prezscore` |
| `dynformer_c1r2/cellnmatched` (and `celllnjw128` on Cartpole/Furuta, the same weights, and the timing twin `cellf_s20260931`) | `xdecpfn/<p>/thesis/default` |
| `dynformer_c1r2/celllnjw128` on PT2 | `xdecpfn/<p>/thesis/default/lnjw128` |
| `dynformer_fr2` | `ftpfn/<p>/thesis/default` |
| `tabpfn/resolution<L>/typeraw`, `stengertpfn/mode<m>`, `stenger` | `tabpfn/<L>`, `segment-tabpfn/<m>`, `segment-gp` |
| `Random uniform+cost`, `RandomESBO`, `Plain GP`, `Stenger ESBO`, `ESBO-TabPFN`, `Oracle BO` | `esrs`, `bo`, `esbo-gp`, `esbo-t-tabpfn`, `esbo-oracle` |
| `MTr 0.9`, `HMTr 0.9` (tau 0.9, crash threshold 0.9) | `decpfn`, `decpfn-h` |
| `MTr 0.99`, `MTr 0.99 ft0.99` (and `HMTr`) | label `decpfn@tau=0.99`, `decpfn@crash_threshold=0.99,tau=0.99` |
| `QCn PI abs 0.99` (`QC PI abs 0.99` on Cartpole/Furuta) | `xdecpfn-pi` |
| `QF EI abs xgrid 0.99` | `ftpfn-ei` |
| `furuta_lqr_rigreset` | `furuta` |

TabPFN `ls`/`diff` encodings, `stengertpfn/modeminimal`, the retired recipe arms and the plain
`furuta_lqr` problem are not imported.

What the local files give (`pfn import thesis-results`, 2026-10-29): `instances` 5,
`prediction_ladder` 7, `prediction_timing` 382, task tables 5, `thesis_series` 898. Raw optplay
runs and raw predplay predictions are cluster-only and are imported by the same command once
copied (`--raw`).

| old file | stored as |
|---|---|
| `optplay/instances/<p>.json` | `instances` (phi, noise and Furuta dead time/kick mapped by position, raw vectors kept) |
| `predplay/tasks/<p>.parquet` | task table `tasks-thesis` (the thesis task selection) |
| `experiments/ladder/results/*.parquet` | `prediction_ladder` (raw per-query cutoff scores, no distributions) |
| `results_timing_bundle/predplay_timing.json` | `prediction_timing` (one call per query on the workstation, as measured) |
| `predplay/results/fig1_cross.csv`, `fig1_per_problem.csv`, `fig1_meta.json`, `fig2_<p>.csv` | `thesis_series` `fig3_cross`, `fig3_per_problem`, `fig3_med_D`, `fig2_stop_curves`, variant `thesis` (the numbers sim_results.tex quotes) |
| the same with `_pre_lcz` | variant `pre_lcz` (before the LC-PFN z-score retraining) |
| `writing/thesis/data/sources/fig1_cross_5prob_canonmed.csv` | `fig3_cross` variant `canonmed` (cluster recomputation, before the retraining) |
| `optplay/results/<p>@g0.05@bcapped_curves.json` | `optplay_finals` (per-instance final incumbent at 15 min(d,4) trials, at `T_budget`, at the cap), `optplay_reference` (J_best over every old cell), `optplay_log_regret_{trials,time}` (unscaled), `optplay_campaign_timing` |
| `..._runs.json`, `..._crashes.json` | `optplay_runs` (per-run summaries), `optplay_crashes` (mean crash count on a 200-point time grid) |
| `<p>@g0.05@bcapped[@timing]/phi_NNNN/runs/<cell>.json` (cluster) | `optimization` / `optimization_timing` with the full `RunRecord` |

### Which figures the local imports rebuild

| figure | from |
|---|---|
| fig:sim:context-sweep | raw (`prediction_ladder`) |
| fig:sim:fig3-cross, fig:app:fig3-sheet, fig:sim:tabpfn-panel | aggregates (`fig3_cross`, `fig3_per_problem` variant `thesis`) with raw timing records |
| fig:sim:lcpfn-decpfn-heads | aggregates (`fig3_per_problem` holds every head and size per problem) |
| fig:sim:ftpfn-xdecpfn-stop-sheet | aggregates (`fig2_stop_curves`) |
| fig:sim:fig2-sheet | aggregates, except the DecPFN-noStates curve (cluster predictions) |
| fig:sim:lcpfn-decpfn-stop-sheet | not locally (only DecPFN and LC-PFN 1M binned curves are in `fig2_*.csv`) |
| fig:sim:optplay-crashes | aggregates (`optplay_crashes`) |
| fig:sim:optplay-violin-cross-full, fig:app:optplay-violin-sheet and the two appendix variant violins | aggregates at 100 % only (`optplay_finals`), the 50 % points need the raw runs |
| fig:sim:optplay-cross, fig:app:optplay-cross-sheet | not locally (needs the per-run incumbent traces, cluster) |
| fig:sim:optplay-timing | not locally (the one-thread timing tree is cluster-only, `optplay_campaign_timing` holds contended campaign times) |
| fig:setup:prior-samples, fig:app:prior-landscapes, tab:app:prior-cost | native units (`priors`, `generation_speed`) |

The local optplay aggregates were written before the last optimization runs (no ESBO-T-TabPFN
rows, some cells on 43-50 instances). `optplay_reference` is the old `J_best` over every cell on
disk. A plot built from the imported finals should recompute `J_best` over its declared set.

## Specification choices

- Cutoffs snap to the last predictable step at or before `f * T` (`floor`). When no predictable
  step lies before it (FT-PFN's coarse grid at 10 %), the first predictable step is used and the
  row is flagged `cutoff_fallback`.
- The stop curves' non-crashed group is the 50 selected non-crashed queries per `(problem, n)`.
- The context of a ladder rung `n` is the first `n` trials of the run (the query is fixed).
- Fields override the budgets so the oracle runs exactly `n_trials` trials.
- `BENCH_SEED = 20261029`. Instance `i` uses `substream(BENCH_SEED, "instances", problem, i,
  attempt)`, field runs `derive_seed(BENCH_SEED, "fields", field, problem, instance, seed)`,
  optimization runs `derive_seed(BENCH_SEED, "optimization", problem, instance, seed)`.
