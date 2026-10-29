# From an empty machine to every figure

This walkthrough runs the whole pipeline: environment, C++ generators, training data, the model
zoo, every benchmark unit, every figure and the rig field. Each step names the command for a
laptop or workstation and the `pfn submit` form for a Slurm cluster (the RWTH CLAIX profile ships
with the repository). Compute figures come from the thesis tables (Borchert 2026, bachelor thesis,
RWTH Aachen): the sampling-cost table tab:app:prior-cost (one core of an Intel Xeon Platinum 8468),
the model-cost table tab:setup:model-cost (one NVIDIA H100) and the optimizer timing figure
fig:sim:optplay-timing. Numbers marked "about" are scaled from those tables to the dataset sizes
and are estimates.

Every command is seeded and resumable. A killed job is submitted again and continues where it
stopped. A finished dataset chunk, model or benchmark unit is never computed twice.

## 0. What runs where

| Step | Resource | Order of magnitude |
|---|---|---|
| install, unit tests, plots, README art | any laptop | minutes |
| training data | many CPU cores | about 2,300 core-hours in total, 2,000 of them for the Furuta FT-PFN contexts |
| model zoo | one CUDA GPU | about 8 GPU hours, plus about 10 for the optional LC-PFN/DecPFN search |
| prediction benchmark | CPU for the fields, a GPU for the predictions | timings belong on the machine you want to report |
| optimization benchmark | one CPU core per run | 2,100 runs, decision time from seconds to hours per run |
| rig field | a Linux workstation with the QUBE-Servo 2 | 6 methods x 5 seeds x 2 rigs, 20 to 55 min per run |

## 1. Install

Prerequisites: [uv](https://docs.astral.sh/uv/), CMake 3.16 or newer, a C++17 compiler and a
LAPACK (Accelerate on macOS, OpenBLAS on Linux, MKL on the RWTH cluster).

```bash
git clone https://github.com/tonyborchert/esbo-pfn.git esbo-pfn && cd esbo-pfn
uv sync
uv run pytest -q                 # unit tests, seconds
uv run pfn doctor                # Python, torch device, compilers, BLAS, TabPFN checkpoint, binaries
uv run pfn config                # resolved PFN_HOME and profile
```

`PFN_HOME` (default `./out`) receives datasets, models, the benchmark database, figures, job
scripts and logs. Point it at a disk with room for the datasets. On Cartpole the 1M
single-trajectory pool takes about 20 GB (2 GB for its first 100k rows), the XDecPFN pools about
33 GB and the FT-PFN contexts about 25 GB. The rest of this page writes `pfn` for `uv run pfn`.

On the cluster, put the profile and the two Slurm accounts into your shell profile once. On RWTH
CLAIX `uv` installs into `~/.local/bin`, the jobs use the `Python/3.12.3` module, and the virtual
environment (about 6 GB with the CUDA wheels) and the uv cache are best kept outside the 150 GB
`$HOME` (the profile copies the two `UV_` variables into every job script):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # once, into ~/.local/bin
module load GCCcore/13.3.0 Python/3.12.3            # before uv sync and every pfn call
export PFN_PROFILE=rwth_claix
export PFN_HOME=$HPCWORK/esbo-pfn-out
export PFN_SLURM_ACCOUNT_CPU=<your cpu project>
export PFN_SLURM_ACCOUNT_GPU=<your gpu project>     # falls back to $PROJECT
export UV_PROJECT_ENVIRONMENT=$WORK/esbo-pfn-venv   # optional, venv outside the checkout
export UV_CACHE_DIR=$WORK/uv-cache                  # optional, uv cache outside $HOME
pfn submit --dry-run -- pfn config                  # inspect the rendered job script
```

The profile and how to add another cluster are described in [CLUSTER.md](CLUSTER.md).

## 2. Build the generators

```bash
pfn data build                                      # macOS, Accelerate
pfn data build --blas openblas                      # Linux
module load GCC/13.3.0 imkl/2024.2.0 && pfn data build --blas mkl    # RWTH, on a login node
uv run pytest --run-cpp                             # C++ against Python parity and distributions
pfn data speed --problem cartpole                   # this machine against the thesis row
```

Binaries are always built on the machine that runs them ([DATA.md](DATA.md#building)).

## 3. TabPFN checkpoint

The TabPFN baselines (Trajectory-TabPFN, Segment-TabPFN, ESBO-T-TabPFN) need the gated TabPFN v3
regressor weights:

```bash
pfn checkpoints download tabpfn                     # license acceptance, TABPFN_TOKEN
pfn checkpoints download tabpfn --from /path/to/tabpfn-v3-regressor-v3_default.ckpt
pfn checkpoints status --verify
```

On a cluster without a browser, download the file elsewhere and use `--from`
([PREDICTORS.md](PREDICTORS.md#checkpoint)).

## 4. Training data

`pfn data gen --for train <model> --problem P` generates what a model needs, `--for all` everything.
Each dataset is written in chunks, and `--shard i/n` (appended by `pfn submit --array`) gives every
array task every n-th chunk of the same directory.

Single-core cost per dataset (training set plus its validation set):

| Problem | `traj/P/1M` (LC-PFN, DecPFN) | `sets/P/xdecpfn` (XDecPFN) | `contexts/P/ftpfn` (FT-PFN) |
|---|---|---|---|
| Boiler | about 30 s | about 1.3 h | about 6 h |
| Three-Tank | about 1 min | about 2.2 h | about 22 h |
| PT2 dead-time | about 1 min | about 4.5 h | about 29 h |
| Cartpole | about 1.3 min | about 5.5 h | about 123 h |
| Furuta | about 1.3 h (twice with the no-dead-time pool) | about 79 h | about 2,000 h |

The thesis table measures 10,000 samples: single trajectories take 0.27 s (Boiler) to 45 s
(Furuta), trajectory sets 1.0 min (Boiler, FT-PFN) to 6.0 h (Furuta, XDecPFN). Furuta dominates
because only 1.5 % of its sampled plants pass the degeneracy screen. Measure your own machine with
`pfn data speed`.

Locally:

```bash
pfn data gen --for train decpfn --problem cartpole --workers 8
pfn data status
pfn data verify traj/cartpole/1M
```

On the cluster (`--kind gen` jobs get 24 cores and the compiler modules):

```bash
for p in boiler threetank_pi pt2_deadtime cartpole furuta
do
  pfn submit --kind gen -- pfn data gen --for train decpfn --problem $p
  pfn submit --kind gen --array 0-9 -- pfn data gen --for train xdecpfn --problem $p
done
pfn submit --kind gen -- pfn data gen --for train decpfn-nodeadtime --problem furuta
for p in boiler threetank_pi pt2_deadtime cartpole
do
  pfn submit --kind gen --array 0-9 -- pfn data gen --for train ftpfn --problem $p
done
pfn submit --kind gen --array 0-99 -- pfn data gen --for train ftpfn --problem furuta
```

The Furuta FT-PFN array above runs 100 tasks of about 20 core-hours each, under one hour per task on
24 cores. Every task that hits its time limit is simply submitted again.

## 5. Model zoo

Training cost on Cartpole on one H100 (tab:setup:model-cost):

| Model | Training data | Parameters | Wall-clock |
|---|---|---|---|
| LC-PFN 100k / 1M (binned) | 10⁵ / 10⁶ trajectories | 5.8 M / 15.3 M | 7.6 s / 4.1 min |
| DecPFN 100k / 1M (binned) | 10⁵ / 10⁶ trajectories | 26.2 M / 7.5 M | 10.9 s / 1.4 min |
| XDecPFN | 132,400 trajectory sets, each reused 256 times | 11 M | 32.1 min |
| FT-PFN | 3,500,000 trajectory sets | 13 M | 41.5 min |

On the other problems XDecPFN and FT-PFN need about the same compute as on Cartpole or less
(tab:app:matched-flops, 0.25 to 1.13 times each model's Cartpole FLOPs). The figures use LC-PFN, DecPFN and
DecPFN-no-St with three heads at two sizes on five problems (90 models, about 2 to 3 GPU hours), the
ten cross-trajectory models (about 5 GPU hours) and two more DecPFN models of the Furuta hardware
ablation.

Locally:

```bash
pfn train decpfn --problem cartpole                         # binned 1M
pfn train lcpfn --problem cartpole --head all --size all    # six models
pfn train all --problem all --head all --size all           # every model, one after the other
pfn models list
```

On the cluster, one job per model family and problem (`--kind train` is one GPU for 2 h):

```bash
for p in boiler threetank_pi pt2_deadtime cartpole furuta
do
  for m in lcpfn decpfn decpfn-nostates
  do
    pfn submit --kind train -- pfn train $m --problem $p --head all --size all
  done
  pfn submit --kind train -- pfn train xdecpfn --problem $p
  pfn submit --kind train -- pfn train ftpfn --problem $p
done
pfn submit --kind train -- pfn train decpfn-nodeadtime --problem furuta
pfn submit --kind train -- pfn train decpfn-nodeadtime-nostates --problem furuta
```

The hyperparameters are the thesis tables in `src/esbopfn/configs/hparams/`. The LC-PFN/DecPFN
search is optional and runs on Cartpole: stage 2 trains 60 cells per size, capped at 25 s (100k) or
250 s (1M) each, about 10 GPU hours for both models and sizes
([TRAINING.md](TRAINING.md#hyperparameter-search)):

```bash
pfn hpo decpfn --problem cartpole --size 1M --stage 1
pfn submit --kind train --array 0-19 -- pfn hpo decpfn --problem cartpole --size 1M --stage 2
```

## 6. Benchmarks

A figure group declares the units it needs. `pfn bench plan <group>` lists the missing ones and
`pfn bench run <group>` computes them in dependency order (`instances`, then `fields`, then the
predictions). Groups share units through the database ([BENCHMARKS.md](BENCHMARKS.md)).

| Group | Needs | Runs on |
|---|---|---|
| `priors` | instances, prior samples, landscape slices | CPU |
| `generation_speed` | one speed unit per problem (C++ binaries) | one core each, minutes |
| `single_trajectory` | `oracle65` fields, predictions of the 90 LC-PFN/DecPFN models, timings | CPU, then GPU |
| `tabpfn` | TabPFN predictions at every context size | GPU (0.4 to 4 s per prediction) |
| `cross_trajectory` | `oracle65` and `oracle181` fields, XDecPFN and FT-PFN predictions | CPU, then GPU |
| `prediction_summary` | the participants of the summary figure | GPU |
| `optimization` | 21 strategy cells x 5 problems x 20 instances, timing units | one CPU core per run |

Locally:

```bash
pfn bench plan prediction_summary
pfn bench run priors
pfn bench run prediction_summary --kind fields
pfn bench run prediction_summary
```

On the cluster, run the stages as array jobs chained with `--after` (each shard takes every n-th
unit of the same sorted list, so shards never collide):

```bash
pfn submit --kind gen --name inst -- pfn bench run prediction_summary cross_trajectory single_trajectory --kind instances
pfn submit --kind gen --array 0-199 --after <inst job> --name field -- \
    pfn bench run prediction_summary cross_trajectory single_trajectory --kind fields
pfn submit --kind bench_gpu --array 0-39 --after <field job> --name pred -- \
    pfn bench run prediction_summary cross_trajectory single_trajectory tabpfn
pfn submit --kind bench --array 0-299 --cpus 1 --name opt -- pfn bench run optimization --kind optimization
pfn submit --kind bench --array 0-104 --cpus 1 --name opt-timing -- pfn bench run optimization --kind optimization_timing
pfn submit --kind bench --name speed --cpus 1 -- pfn bench run generation_speed
```

Optimizer decision time per run on one core (fig:sim:optplay-timing, Boiler to Furuta) ranges from
3 to 10 s for BO, 5 to 174 s for ESBO-GP, 8 to 34 s for DecPFN, 17 to 196 s for DecPFN-H, 20 to
142 s for XDecPFN-PI and 31 to 354 s for FT-PFN-EI, and from 2 min to 4 h for ESBO-T-TabPFN. The
simulation of the trials comes on top. Inference timings (`prediction_timing`) should run on the
machine whose times you want to report, the thesis used a workstation with a Quadro RTX 4000:

```bash
pfn bench run prediction_summary --kind prediction_timing
```

## 7. Figures

```bash
pfn plot --list
pfn plot                                    # every group, PDF into $PFN_HOME/figures/<group>/
pfn plot hardware --format png --theme black
pfn plot readme --out /tmp/art              # the README images
```

Each figure gets a sidecar `<name>.md` with what it shows, the metric, the data units and the numbers
it is drawn from. A figure whose units are missing is skipped with its reason in `skipped.txt`.
Three figures read trained models directly (the LC-PFN/DecPFN example, the FT-PFN/XDecPFN example and
the Furuta landscape), and the search panel reads the HPO results ([FIGURES.md](FIGURES.md)).

## 8. Hardware

The rig needs the Quanser HIL SDK and BlueRiverTech's `gym_brt` on the workstation, plus the models
`decpfn/furuta/binned/1M` and `xdecpfn/furuta/default/default` (and the four DecPFN models of the
ablation for `--ablation`). Setup, reset and safety nets are in [HARDWARE.md](HARDWARE.md).

```bash
pfn hw serve --plant qube --port 8765                # one server, on the workstation
pfn hw run --rig without_weight                     # 6 methods x seeds 0-4, N_max 180, T_budget 36000
pfn hw run --rig with_weight                        # after mounting the 5 g tip mass
pfn hw status --rig with_weight
pfn hw bench-pred --rig without_weight              # Segment-GP, DecPFN, XDecPFN ladder
pfn hw bench-pred --rig without_weight --ablation   # the four DecPFN models
pfn plot hardware
```

`pfn hw run --rig without_weight --plant sim` runs the same field against a simulated rig in
process, which is also how the stack is tested without hardware. A physically different rig is a new
entry in `src/esbopfn/configs/rigs.yaml`.

## 9. Thesis artifacts

Where the thesis results, rig logs and checkpoints are available, the importers load them into the
same database and model tree with `provenance: thesis`, and every figure group then draws from them
next to new units:

```bash
pfn import thesis-results --from <thesis project> [--raw <optplay trees>]
pfn import thesis-models --from <checkpoint dir>
pfn hw import-logs <run tree>
pfn hw import-pred <hwpred results dir>
```

What each import provides and which figure it serves is listed in
[BENCHMARKS.md](BENCHMARKS.md#importing-the-thesis-results),
[CROSS_TRAJECTORY.md](CROSS_TRAJECTORY.md#thesis-checkpoints), [LEGACY_MODELS.md](LEGACY_MODELS.md) and
[HARDWARE.md](HARDWARE.md#importing-the-thesis-logs).
