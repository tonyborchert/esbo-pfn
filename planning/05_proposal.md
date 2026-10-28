# 05 — Proposal for the clean repository (`../new`)

Status: DRAFT for Tony's review, 2026-10-28. Built on the four audits `01`–`04` in this folder.
Nothing here is implemented yet. Every open decision is collected in Section 9.

---

## 1. Goal and principles

One repository that reproduces the thesis end to end: five problems, data generators, four own
predictors plus the off-the-shelf baselines, all optimization strategies, the rig interface, every
benchmark and every thesis figure, on the RWTH cluster or anywhere else.

Principles:

1. **The thesis text is the specification.** Where the old code deviates from the thesis, the new
   code follows the thesis (default). Exceptions are decided explicitly in Section 9 and recorded in
   `docs/DEVIATIONS.md` (thesis statement, what the code does, why).
2. **Everything is seeded and idempotent.** Data generation, training, benchmark runs and plots are
   deterministic given their config and seed, and re-running a finished step is a no-op.
3. **One tree of outputs, one env var.** `PFN_HOME` holds datasets, models, benchmark database,
   figures and logs. Default `./out` locally, `$HPCWORK/<repo>` on the cluster.
4. **One submission layer.** Every CLI command can run locally or be wrapped into a Slurm job by
   `submit`, driven by a cluster profile file. Nothing cluster-specific lives in the science code.
5. **Old thesis artifacts are first-class data.** Checkpoints, benchmark results and rig logs from the
   thesis are importable into the same database and model tree, so the README can be built from
   them, and new runs land next to them.
6. **Tests at every layer**, marked by what they need (`unit`, `cpp`, `gpu`, `slurm`, `hardware`).

---

## 2. Repository layout

```
new/                                   # the published repo (name: Q-A1)
├── README.md                          # results summary (Section 8)
├── pyproject.toml                     # uv project, python 3.12, one package `<pkg>` (Q-A1)
├── uv.lock
├── LICENSE                            # (Q-A3)
├── configs/
│   ├── hparams/                       # COMMITTED canonical hyperparameters (thesis tables)
│   │   ├── lcpfn.yaml                 # 6 rows: {100k,1M} × {binned,gaussian,linear}; also used by lcpfn-last
│   │   ├── decpfn.yaml                # 6 rows; also used by decpfn-nostates
│   │   ├── xdecpfn.yaml               # one config (+ per-problem steps/stride)
│   │   └── ftpfn.yaml
│   ├── problems.yaml                  # T, dt, stride, N_grid, budgets per problem (thesis tab:app:*)
│   ├── datasets.yaml                  # named training sets and their sizes (Section 5)
│   ├── benchmarks/                    # plot-group specs (which benchmarks × problems × methods)
│   └── cluster/
│       ├── local.yaml                 # no Slurm, run in place
│       └── rwth_claix.yaml            # accounts, partitions, modules, roots, defaults per job kind
├── src/<pkg>/
│   ├── problems/                      # boiler, threetank, pt2_deadtime, cartpole, furuta
│   │   ├── base.py                    # System, Objective, Trial, crash/success, grids
│   │   ├── prior.py                   # the universal prior recipe (Scale/Bounded/Tied, noise, φ̂, screen)
│   │   └── <problem>/{system,controller,objective}.py
│   ├── generators/                    # Python wrappers around cpp/ binaries + pure-Python fallback
│   │   ├── pools.py                   # pool format (memmap float32 + spec json), reservation, sizes
│   │   ├── trajectories.py            # single-trajectory prior (LC-PFN/DecPFN data)
│   │   ├── sets.py                    # trajectory-set priors (XDecPFN pools, FT-PFN contexts)
│   │   └── speed.py                   # generation-speed benchmark (tab:app:prior-cost)
│   ├── predictors/
│   │   ├── base.py                    # Predictor, Prediction (bar / gaussian / atom), bins
│   │   ├── lcpfn/                     # vendored released LC-PFN, crash-aware head, `last` variant
│   │   ├── decpfn/
│   │   ├── xdecpfn/                   # flattened from dynformer c1r2/i5/s12/f3
│   │   ├── ftpfn/                     # flattened from dynformer fr2
│   │   ├── tabpfn.py                  # Trajectory-TabPFN (L ∈ {25,50,100,250})
│   │   ├── segment_tabpfn.py          # split / rate
│   │   └── segment_gp.py              # ESBO-GP section GPs, matlab_gp port
│   ├── training/
│   │   ├── registry.py                # model ids ↔ paths, ModelNotTrained errors
│   │   ├── train_{lcpfn,decpfn,xdecpfn,ftpfn}.py
│   │   ├── hpo.py                     # staged search (greedy lr/wd/dropout, arch grid, Pareto plots)
│   │   └── common.py                  # AdamW, cosine+warmup, bf16, clip, seeding, resume
│   ├── strategies/
│   │   ├── base.py                    # Optimizer ABC, budgets, initial design, incumbent
│   │   ├── gp.py                      # objective GP, MES, candidate sets
│   │   ├── esrs.py, bo.py, esbo_gp.py, esbo_tabpfn.py, oracle.py
│   │   ├── rules.py                   # exceed / naive / matched rules, τ, τ_crash
│   │   ├── decpfn_esbo.py             # DecPFN, DecPFN-H (virtual observations)
│   │   └── pure_loop.py               # XDecPFN-PI / FT-PFN-EI (closed-form EI/PI on bins)
│   ├── hardware/
│   │   ├── client.py                  # websocket client to the rig server
│   │   ├── server/                    # remote_env_server + gym_brt driver (Q-C1)
│   │   ├── reset.py                   # alg:app:hw-reset
│   │   ├── runner.py                  # rig field (methods × seeds), resumable trials.jsonl
│   │   └── prediction_bench.py        # hwpred ladder + ablation
│   ├── benchmarks/
│   │   ├── db.py                      # SQLite + blob store, spec hashing, "exists?" queries
│   │   ├── runner.py                  # run a benchmark spec (local / shard / array)
│   │   ├── prediction.py              # oracle fields → tasks → predictions → scores
│   │   ├── optimization.py            # strategies × instances × seeds
│   │   ├── timing.py                  # inference + optimizer timing
│   │   ├── landscapes.py              # prior landscapes, prior samples
│   │   └── import_thesis.py           # import old results/checkpoints/rig logs into the DB
│   ├── plots/
│   │   ├── style.py                   # thesis style + method colors + RWTH palette
│   │   ├── <figure_group>.py          # one module per thesis figure group (Section 7)
│   │   └── readme_art.py              # 3D hero figure and README graphics
│   ├── cluster/
│   │   ├── profile.py, submit.py      # job rendering + sbatch / local
│   │   └── templates/job.sh.j2
│   └── cli.py                         # `pfn` entry point
├── cpp/                               # the C++ generators (furuta, cartpole, plants) + gen for all
│   ├── CMakeLists.txt                 # MKL / OpenBLAS / Accelerate selection (Q-B5)
│   └── ...
├── tests/
│   ├── unit/                          # pure python, seconds, Mac
│   ├── cpp/                           # needs built binaries (parity C++ vs Python)
│   ├── gpu/                           # tiny trainings on cuda/mps
│   ├── slurm/                         # submit --dry-run, profile rendering, one real tiny job
│   └── hardware/                      # rig server round trip (skipped without rig)
├── thesis_artifacts/                  # small committed data (Q-B7): rig logs, result CSVs, manifests
├── docs/                              # DEVIATIONS.md, CLUSTER.md, HARDWARE.md, FIGURES.md
└── assets/                            # README figures (png/svg)
```

The current `new/planning/` folder is removed (or moved to `docs/planning/`) before publishing.

---

## 3. Output tree (`PFN_HOME`)

```
$PFN_HOME/
├── datasets/<problem>/<dataset>/          # pool.npy_raw + pool.spec.json + meta.json (seed, binary hash)
├── models/<model>/<problem>/<variant>/<size>/<run>/   # model.pt, config.yaml, metrics.json, calib.json
├── hpo/<model>/<problem>/<size>/          # trials.csv, candidate hparams yaml, gridsearch figures, models
├── checkpoints/                           # downloaded third-party weights (TabPFN v3)
├── benchmarks/
│   ├── db.sqlite                          # index: spec hash, status, paths, timings, code version
│   └── blobs/<kind>/<hash>/               # run json, predictions parquet, trials.jsonl
├── figures/<group>/<name>.pdf             # default white pdf; --format png / --theme black
├── jobs/                                  # rendered Slurm scripts
└── logs/                                  # job stdout/err
```

- Model ids (used by strategies, benchmarks and plots):
  `lcpfn/<problem>/<head>/<size>`, `lcpfn-last/...`, `decpfn/...`, `decpfn-nostates/...`,
  `xdecpfn/<problem>`, `ftpfn/<problem>`, `tabpfn/<L>`, `segment-tabpfn/<split|rate>`, `segment-gp`.
  Default `<head>=binned`, `<size>=1M`. `<run>` is `0` unless several seeds are trained.
- A strategy that needs a model raises `ModelNotTrained("decpfn/cartpole/binned/1M", hint="pfn train decpfn --problem cartpole")`.
  TabPFN raises `CheckpointMissing` with the download instruction.
- Thesis checkpoints are imported with `pfn import thesis-models <archive>` into the same tree with
  `provenance: thesis` in `config.yaml`.

---

## 4. Cluster / Slurm layer

Profile `configs/cluster/rwth_claix.yaml` (everything cluster-specific lives here):

```yaml
name: rwth_claix
root: ${HPCWORK}/pfn                       # PFN_HOME on this cluster
scheduler: slurm                           # or: none
accounts: {cpu: rwth2055, gpu: thes2287}
partitions: {cpu: c23ms, gpu: c23g}
modules: [GCCcore/13.3.0, Python/3.12.3]
modules_cpp: [GCC/13.3.0, imkl/2024.2.0]
pre: |                                     # shell lines before the command
  export PATH="$HOME/.local/bin:$PATH"
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python: uv run --no-sync python
defaults:                                  # per job kind
  gen:   {kind: cpu, cpus: 24, mem: 16G, time: "04:00:00"}
  train: {kind: gpu, gpus: 1, cpus: 8, mem: 64G, time: "02:00:00"}
  bench: {kind: cpu, cpus: 8,  mem: 16G, time: "04:00:00"}
  bench_gpu: {kind: gpu, gpus: 1, cpus: 8, mem: 64G, time: "02:00:00"}
gpu_select: uuid                           # login node: pick full H100 by UUID, never ordinal
```

CLI:

```
pfn submit [--profile rwth_claix] [--kind train|gen|bench] [--time ..] [--cpus ..] [--array 0-19]
           [--after JOBID] [--dry-run] [--local] -- <any pfn command>
```

- Renders `$PFN_HOME/jobs/<ts>_<name>.sh` from the template (header, modules, pre, cd, command),
  then `sbatch`. `--local` runs the command in place (Mac, workstation, a cluster without Slurm).
  `--array` maps `$SLURM_ARRAY_TASK_ID` onto the command's `--shard i/N` or `--instance`.
- Every `pfn` command is resumable, so a killed job (30-min login-node limit, time limit) is just
  resubmitted. `pfn train ... --until-done` loops the command locally until it reports completion
  (the login-node H100 path used for the thesis models, Q-B3).
- `pfn status` lists datasets, models, benchmark completion from the DB, and Slurm queue state.
- Another cluster = copy the profile and edit accounts, partitions, modules, root.

---

## 5. Data generation and training interface

`configs/datasets.yaml` declares every training set the thesis uses:

| dataset | prior | size (thesis) | consumer |
|---|---|---|---|
| `traj/<problem>/1M` | single-trajectory | 1,000,000 (+ 1,024 val) | LC-PFN, DecPFN 1M (and variants) |
| `traj/<problem>/100k` | single-trajectory | 100,000 (+ val) | 100k models (prefix of 1M pool) |
| `sets/<problem>/xdecpfn` | screened system pools of N_max full trajectories | 132,400 (+256 held-out) | XDecPFN |
| `contexts/<problem>/ftpfn` | trajectory-set prior, fresh per example | 3,500,000 (+2,000 held-out) | FT-PFN |
| `fields/<problem>/oracle65`, `oracle181` | ESBO-Oracle runs on screened instances | per thesis | prediction benchmark, context ladder |

```
pfn data status [--problem P]                 # what exists, what each training config needs
pfn data gen traj --problem cartpole --size 1M [--workers 24] [--seed 0]
pfn data gen --for train xdecpfn --problem furuta    # resolves the needed dataset(s)
pfn data gen --for all                         # everything the thesis trained on
pfn data verify traj/cartpole/1M               # C++ ↔ Python parity + invariants
```

Generation is C++ for all five problems (`plants` gets a `gen` subcommand, Q-D10), seeded, chunked
(`--workers` does not change the result: each chunk has its own seed stream), resumable (pool grows
to the target count). The Python simulators remain for benchmarks and parity tests.

Training:

```
pfn train decpfn --problem cartpole [--head binned] [--size 1M] [--hparams configs/hparams/decpfn.yaml]
pfn train lcpfn-last --problem furuta
pfn train xdecpfn --problem pt2_deadtime
pfn train ftpfn   --problem boiler
pfn train all --problem all                     # the full thesis model zoo
pfn hpo decpfn --problem cartpole --size 1M     # → $PFN_HOME/hpo/..., writes decpfn.yaml candidate + figures
```

- `--hparams` points at the committed file by default; an HPO output file can be passed instead.
- Trainer writes `config.yaml` (resolved hparams + dataset + seed + git hash), checkpoints every N
  steps, `metrics.json`, `calib.json`, and resumes automatically.
- Variants are flags on the same trainer: `lcpfn-last` = LC-PFN with the target set restricted to
  the final grid point, `decpfn-nostates` = DecPFN with `(J, ΔJ, t)` tokens only.

---

## 6. Benchmarks and the database

Two layers, as requested:

**Runner** (`pfn bench run <kind> ...`) executes atomic units and stores results in the DB keyed by a
spec hash (kind, problem, instance/seed, method/model spec incl. checkpoint hash, budget, code
version). Already-present units are skipped. Units are shardable so a Slurm array can split them.

| kind | unit | needs |
|---|---|---|
| `fields` | ESBO-Oracle run on (problem, instance, seed, trials) | CPU |
| `prediction` | predictions of one model on one problem's task table | CPU or GPU |
| `prediction-ladder` | one model × rung × problem | CPU/GPU |
| `prediction-timing` | one model × problem × n, repeated calls | one GPU (workstation) |
| `optimization` | one strategy × problem × instance (× seed) | CPU (ESBO-TabPFN: long) |
| `optimization-timing` | one strategy × problem × instance, 1 core | CPU |
| `generation-speed` | one problem × generator mode, 1 core | CPU |
| `landscapes` | one problem × instance × slice | CPU |
| `hardware-optimization` | one cell × seed on one rig config | the rig, run separately per rig config |
| `hardware-prediction` | one model × ladder on one rig log tree | CPU/GPU |

**Plot groups** (`configs/benchmarks/<group>.yaml` + `src/<pkg>/plots/<group>.py`) declare which
units they need (problems × methods × seeds) and render figures. `pfn bench plan <group>` lists
missing units, `pfn bench run <group>` runs them, `pfn plot <group>` renders. Groups overlap freely
through the DB.

Output: `$PFN_HOME/figures/<group>/<name>.pdf`, white. `pfn plot --format png --theme black` for the
alternatives, no config edit.

Import of the thesis results: `pfn import thesis-results <archive>` fills the DB from the old
optplay/predplay/hwpred/hwplay/timing files so every thesis figure can be plotted immediately and
the README is built from the thesis numbers.

---

## 7. Thesis figure groups (what `pfn plot` must cover)

| group | figures (thesis) |
|---|---|
| `priors` | prior samples, prior landscapes |
| `single_trajectory` | LC-PFN/DecPFN example, heads benchmark sheet, stop sheet, HPO gridsearch panels |
| `tabpfn` | TabPFN panel (Trajectory/Segment TabPFN Pareto) |
| `cross_trajectory` | context ladder, FT-PFN/XDecPFN example, Furuta landscape sheet, stop sheet |
| `prediction_summary` | fig3 cross, fig3 sheet, fig2 sheet |
| `optimization` | optplay cross, cross sheet, violins, crashes, timing, DecPFN threshold violins, qx variants |
| `hardware` | optimization (with/without), regret violin, cost first 1000 s, best controller at 500 s, acquisition wait, hwpred benchmark, hwpred ablation |
| `generation_speed` | prior-cost table (as a markdown/LaTeX table) |
| `readme` | 3D hero figure, summary cards |

---

## 8. README plan

1. One-paragraph pitch, hero figure (3D: real rig cost trajectories, XDecPFN predicted total-cost
   bands over θ, next controller, crash probability; rendered with matplotlib from real rig data and
   the real Furuta XDecPFN checkpoint, Q-E1).
2. The five problems (table + prior samples figure).
3. Predictors: XDecPFN and FT-PFN explained with the thesis schematics, then DecPFN/LC-PFN/baselines.
4. Results: prediction summary (fig3 cross), optimization cross, hardware optimization + the
   500 s controllers, the hardware prediction benchmark. Numbers from the thesis.
5. Reproduce: install, data, train, benchmark, plot, cluster profile, hardware.
6. Model zoo and artifact download, citation, license.

Writing rules carry over (AmE, no semicolons, "early stopping" unhyphenated).

---

## 9. Decisions needed (asked in chat, recorded here)

Groups: A repo/infra, B outputs/cluster, C hardware, D thesis-vs-code fidelity, E README/figures,
F what to fetch from the cluster now, G scope confirmations.

(See the chat message of 2026-10-28 for the full list with recommendations. Answers will be recorded
in `06_decisions.md`.)

---

## 10. Build order (after decisions)

1. Skeleton, pyproject, cluster profiles, `submit`, DB, CLI, test harness.
2. Problems + prior + Python simulators + tests against old code (golden trajectories).
3. C++ generators (port, `gen` for plants, CMake, LAPACK fallback) + parity tests.
4. Pool format, datasets, `data gen/status`.
5. Predictors (base, DecPFN, LC-PFN, XDecPFN, FT-PFN, TabPFN, Segment-GP) + load-old-checkpoint tests.
6. Trainers + HPO + hparams files + GPU smoke tests.
7. Strategies + rules + tests on toy problems.
8. Benchmarks: fields, prediction, optimization, timing, landscapes, generation speed.
9. Thesis import, plots, figure parity with the thesis PDFs.
10. Hardware: client/server, runner, hwpred, dry-run tests via the server's simulator.
11. README, docs, DEVIATIONS.md, release packaging.
