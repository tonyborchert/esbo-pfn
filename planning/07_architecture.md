# 07 — Architecture and interface spec for the build agents

Binding for every implementation agent. Read `01_thesis_inventory.md` (the spec of what to
reproduce), `06_decisions.md` (decisions), then this file. The audits `02`–`04` tell you where the
old code is when you need to port numerics. Old code is a reference for numerics only, never copy
structure, env-var plumbing or `sys.path` hacks.

Repo root: `/Users/tonyborchert/Documents/Uni/bachelorarbeit/new`. Old code:
`/Users/tonyborchert/Documents/Uni/bachelorarbeit/impl/{prediction,dynformer}`. Thesis:
`/Users/tonyborchert/Documents/Uni/bachelorarbeit/writing/thesis/sections/*.tex`.

## 0. Conventions

- Python 3.12, `uv` project, package `esbopfn` under `src/`, console script `pfn`.
- Type hints everywhere, dataclasses for configs/records, numpy + torch, `logging` not `print`.
- No module-level side effects, no import-time environment binding, no `sys.path` edits.
  Configuration reaches code through explicit arguments. Only two env vars exist:
  `PFN_HOME` (output root) and `PFN_PROFILE` (cluster profile name, default `local`).
- Small modules. A trainer file is one screen of logic on top of `training/common.py`. The four own
  predictors follow the same file layout: `model.py` (nn.Module + config dataclass), `data.py`
  (dataset → batches), `train.py` (one `train(cfg) -> ModelDir` function), `predictor.py`
  (inference adapter implementing `Predictor`).
- CLI: `src/esbopfn/cli.py` discovers `src/esbopfn/commands/*.py`, each exposing
  `register(subparsers)`. **An agent adds its command by adding a file, never by editing
  `cli.py`.** Same rule for plot groups (`plots/<group>.py` exposing `GROUP = PlotGroup(...)`) and
  benchmark kinds (`benchmarks/kinds/<kind>.py` exposing `KIND = BenchmarkKind(...)`).
- Every random draw takes an explicit `numpy.random.Generator` or torch generator derived from a
  seed through `esbopfn.util.seeds.substream(root_seed, *labels)` (SeedSequence based). No global
  RNG use anywhere.
- Tests: pytest under `tests/`, markers `unit` (default), `cpp`, `gpu`, `slurm`, `hardware`
  (configured in `pyproject.toml`, non-unit markers skip unless `--run-<marker>` or the resource is
  detected). Unit tests must run on this Mac in seconds with the Mac venv.
- Writing: AmE, no semicolons in docs and docstrings, "early stopping" unhyphenated.
- Never write anything about the thesis results being regenerated or replaced. Code documents what
  it does. `docs/DEVIATIONS.md` lists remaining differences from the thesis text (there should be
  few: see §11).

## 1. Core data model (`esbopfn/core.py`)

```python
@dataclass(frozen=True)
class ProblemSpec:          # from configs/problems.yaml + the problem package
    name: str               # boiler | threetank_pi | pt2_deadtime | cartpole | furuta
    d: int
    theta_lo: np.ndarray; theta_hi: np.ndarray
    T: int                  # logged steps (T_max)
    channels: tuple[str, ...]        # level channels ξ (what predictors see, thesis tab:app:problems)
    rate_channels: tuple[str, ...]   # measured rates ξ̇ (empty → ξ̇ = Δξ)
    can_crash: bool
    has_success: bool
    def budget_d(self): return min(self.d, 4)
    def n_max(self): return 3 * 15 * self.budget_d()
    def t_budget(self): return 15 * self.T * self.budget_d()
    def n_init(self): return max(self.d, 2)

@dataclass
class Instance:             # one sampled plant (thesis §1.3 universal prior)
    phi: dict[str, float]; phi_hat: dict[str, float]; noise_std: dict[str, float]
    extras: dict[str, float]   # furuta dead_time, kick; empty elsewhere
    seed: int                  # instance id used to derive per-trial seeds

@dataclass
class Trajectory:           # full simulation output, uncut
    theta: np.ndarray
    J: np.ndarray           # (T,) float32 cumulative cost, +inf from the crash step on
    x: np.ndarray           # (T, C) level channels as the predictors see them (noisy)
    xdot: np.ndarray | None # (T, C_rate) measured rates, None if none
    crash_step: int | None  # 1-based step of the crash, None if no crash
    success: bool

@dataclass
class Trial:                # what an optimizer / predictor sees: a trajectory observed up to t_obs
    traj: Trajectory
    t_obs: int              # 1..T observed steps (early stopped if < T and not crashed)
    @property def crashed(self): ...     # crash_step is not None and crash_step <= t_obs
    @property def complete(self): ...    # t_obs == T and not crashed
    @property def J_obs(self): ...       # J[t_obs-1] (inf if crashed)
```

Grids (`esbopfn/grids.py`), all 1-based steps as in the thesis:
- `stride(spec)`: reference cartpole stride 15, `N_ref = ceil(1000/15) = 67`,
  `stride = argmin_s |ceil(T/s) − N_ref|`, ties → smaller s. `n_grid(spec) = ceil(T/stride)`.
- `decoder_grid(spec)`: `t_l = l·stride, l = 1..N_grid−1`, plus `T` as the horizon (not a token).
- `curve_grid(spec, N)`: `t_l = floor(l·T/N), l = 1..N` (LC-PFN with N = n_grid, TabPFN with N = L,
  FT-PFN with N = 8).
- `tokens(traj, grid, t_obs)` → covariates `c_l = [J_{t_l} − J_{t_{l−1}}, ξ_{t_l}, Σ_{t_{l−1}<t≤t_l} ξ̇_t]`
  with `t_0 = 0, J_0 = 0` for every grid point `t_l ≤ t_obs`. Used by DecPFN and XDecPFN.

Predictions (`esbopfn/predictors/base.py`):
```python
@dataclass
class BarPrediction:       # crash-aware bar distribution over the total cost J_T
    edges: np.ndarray      # (N_bin+1,) raw cost units, increasing
    probs: np.ndarray      # (N_bin,) sums to 1 − p_crash
    p_crash: float
    def log_density(self, y): ...; def cdf(self, y): ...; def conditioned_above(self, j_obs): ...
    def sample(self, rng, n): ...; def mean_var(self): ...
@dataclass
class GaussianPrediction: mean: float; var: float; p_crash: float   # same methods

class Predictor(Protocol):
    name: str
    def predictable_steps(self, spec) -> np.ndarray         # steps at which a prediction is defined
    def predict(self, query: Trial, context: Sequence[Trial]) -> Prediction
    def predict_many(self, queries, context) -> list[Prediction]   # default loops
```
Context-blind predictors ignore `context`. `query.t_obs == 0` means an unrun θ (allowed for the
cross-trajectory models and the GP/TabPFN baselines).

Optimizers (`esbopfn/strategies/base.py`):
```python
class Strategy(ABC):
    def __init__(self, spec, rng, **params)
    def propose(self, history: list[Trial]) -> np.ndarray        # θ_{n+1}
    def should_stop(self, running: Trial, history) -> bool        # called after each logged step
    def incumbent(self, history) -> tuple[np.ndarray, float]     # θ⁻, J⁻ (thesis: best successful complete trial)
```
The benchmark loop (`strategies/loop.py`) owns budgets (`n_max`, `t_budget`, initial design `n_init`
uniform without stopping), the simulator, seeds and recording. Strategies never simulate.

## 2. Problems (`esbopfn/problems/`)

- One subpackage per problem with `simulate(theta, instance, rng, record_rates) -> Trajectory`
  and the module-level `SPEC: ProblemSpec` and `PRIOR: PriorSpec`. Pure numpy, vectorization not
  required but keep inner loops tight (the C++ generators are the fast path).
- `prior.py`: `PriorSpec` = list of `ScaleParam(name, nominal, f_lo, f_hi)`, `BoundedParam(name, lo, hi)`,
  `TiedParam(name, source, zeta=1.5)`, noise bases per channel, `estimate_sigma_rel = 0.10`,
  `uses_phi_hat: bool`. Functions `sample_instance(prior, rng)`, `sample_theta(spec, rng)`,
  `screen(problem, instance, rng, n_theta=100, repeats=10, min_pass=25)`,
  `single_trajectory_prior(problem, rng, max_unsuccessful_share=0.5)`,
  `trajectory_set_prior(problem, instance, n, rng)` exactly as thesis §1.3 (M:409-466).
  Noise: `σ_i = base_i · S · η_i`, `S ~ LU(1.5/4, 4/1.5)` per instance, `η_i ~ LU(1/1.5, 1.5)` per
  channel, for every problem. `p̂ = max(0, N(p, (p/10)²))`.
- `problems.yaml` holds the numbers of the thesis tables (A:597-858), the code reads them.
- Furuta: dataset = realistic rig reset (dead time U[8.33, 25] ms per instance, kick U[0, 0.6] V per
  instance with random sign per trial, 30 ms pulse), prior flag `dead_time: false` variant for the
  hardware ablation. Success = mean |φ_pend| over the last 2 s < 5°. Cost `|φ_pend|(1 + 20(1 − min(|φ_pend|, 5°)/5°))`.

## 3. Generators and pools (`esbopfn/generators/`, `cpp/`)

- C++ (`cpp/`): one binary per problem family (`furuta_gen`, `cartpole_gen`, `plants_gen`) with
  subcommands `traj` (single-trajectory prior → pool), `sets` (trajectory-set prior → set pool),
  `screen`, `rollout` (one θ on one instance, for parity tests), `speed` (generation-speed bench).
  Same RNG contract as Python at the distribution level, not bit level. Seeded per chunk:
  `chunk_seed = substream(seed, chunk_index)`, so the result is independent of the worker count.
  CMake with `-DBLAS=mkl|openblas|accelerate`.
- Pool format (`generators/pools.py`): directory `<PFN_HOME>/datasets/<problem>/<dataset>/` with
  `rows.f32` (memmap `(N, R)` float32), `spec.json` (column layout: theta d, instance fields,
  `crash_step`, `success`, `J[T]`, `x[T·C]`, `xdot[T·C_r]`), `meta.json` (seed, count, generator
  version, problem spec hash, chunk size). `Pool.open(path)`, `Pool.read(i) -> Trajectory`,
  `Pool.slice(a, b)`. Growth by chunks, `pool.finished` marker.
- Set pools (XDecPFN): `instances.json` + rows tagged with instance index, N_max full trajectories
  per instance. FT-PFN contexts: pre-tokenized on the FT-PFN grid (8 points + last step + θ) in
  shards, since full trajectories are too large (thesis: 3.5 M fresh sets).
- `configs/datasets.yaml` names: `traj/<p>/100k`, `traj/<p>/1M` (100k is the first 100,000 rows of
  the 1M pool, validation = a separate held-out seed stream), `sets/<p>/xdecpfn` (132,400 + 256),
  `contexts/<p>/ftpfn` (3,500,000 + 2,000), `fields/<p>/oracle65`, `fields/<p>/oracle181`.
- `pfn data status|gen|verify` as in `05_proposal.md` §5.

## 4. Predictors (`esbopfn/predictors/`)

Own models, all with the crash-aware heads of thesis M:553-613 (`heads.py`: `BarHead` binned /
linear edges, `GaussianHead`, shared NLL, N_bin = 1000, edges from prior samples, out-of-support
clamped), input z-scoring from prior samples (`normalization.py`), time z-scored over its range:
- `lcpfn/`: encoder, post-LN, two-layer FFN head, tokens `(t_l, J_{t_l})` on `curve_grid`, one
  cutoff κ ~ U{1..N−1} per batch, queries = remaining grid points, crashed-before-κ rows removed
  from the batch (no requeue). Variant `last`: queries = only the final grid point.
- `decpfn/`: causal pre-LN decoder, tokens on `decoder_grid` with covariates `tokens()`, head on
  every position predicting `J_T`, only cutoffs before a crash scored, loss = mean over valid
  positions. Variant `nostates`: covariates `(J, ΔJ, t)`.
- `xdecpfn/`: stage 1 = DecPFN stack per trajectory (cacheable), token for each trajectory =
  final stage-1 token + e_θ(θ) + e_{complete|cut|crash}, unrun queries = e_θ(θ*) + e_unrun, stage 2
  bidirectional self-attention, mask: no token attends to query tokens, head on trajectory tokens
  (horizon) and query tokens. Target ln J, bins equal-mass in ln J, mapped back to J at inference.
  Batch = N_sys × N_reuse examples, N_unrun = 8, loss = mean over all predictions (M:1057-1132).
- `ftpfn/`: PFN-style encoder with per-context block attention, tokens
  `((θ_i, t_l), J_{t_l})` on the 8-point `curve_grid` for `t_l ≤ T_i` plus `((θ_i, T_i), J_{T_i})`
  only if the trial was truncated, crash embedded as learned `e_∞`, queries at the horizon for
  truncated trials + N_unrun = 16 unrun θ, raw J target, loss mean within example then over batch.
- `tabpfn.py` (Trajectory-TabPFN, L ∈ {25,50,100,250}, TabPFN v3 regressor default params),
  `segment_tabpfn.py` (split, rate), `segment_gp.py` (ESBO-GP section GPs on `gp.py`'s exact GP:
  constant mean, ARD squared-exponential, log hyperparameters, Adam, MES; `cold_refit_growth`
  parameter). TabPFN checkpoint path `<PFN_HOME>/checkpoints/tabpfn/`, `CheckpointMissing` error
  with download instructions.
- Hyperparameters: `configs/hparams/<model>.yaml`, exactly the thesis tables (A:497-517, A:545-559).
  Shared training core `training/common.py`: AdamW, linear warmup 25% from 0 to lr then cosine to 0
  stepped per optimizer step, grad-norm clip 1, bf16 autocast on CUDA, batch 128 for LC-PFN/DecPFN,
  heads = d_model/16, d_ff = 2·d_model, seeded init, checkpoint + resume, `metrics.json`,
  optional wall-clock cap (HPO).
- Model directory: `<PFN_HOME>/models/<model>/<problem>/<variant>/<size>/<run>/` with `model.pt`
  (state_dict + config), `config.yaml`, `calib.json` (bin edges, normalization stats), `metrics.json`.
  Model id string `<model>/<problem>/<variant>/<size>[/<run>]`, resolved by `training/registry.py`,
  `ModelNotTrained(id, hint)` when absent.

## 5. Strategies (`esbopfn/strategies/`)

Thesis §4 exactly: ESRS, BO, ESBO-GP (Segment-GPs, virtual observations resampled every
iteration, crashed virtual objective formula), ESBO-Oracle, ESBO-T-TabPFN (L = 50), DecPFN and
DecPFN-H with `exceed` + `matched` rule (τ default 0.9, incumbent predictions reused from its own
run, sampled virtual observations conditioned on `J ≥ J_{T_i}`, hybrid picks the lower-variance
source per trial per iteration), pure loops XDecPFN-{PI,EI} and FT-PFN-{EI,PI} (`naive` rule τ
default 0.99, closed-form EI/PI on bins in raw J, candidate set 400 uniform + 1600 Gaussian
σ = 0.10 × extent around the top-5 completed θ, FT-PFN stop decisions on the decoder grid).
Optional `crash_threshold` (None by default) only to reproduce the appendix τ_crash variants.
Objective GP for BO/ESBO: the port of Stenger's exact GP (`predictors/gp.py`), MES acquisition
over 2000 uniform random candidates, `cold_refit_growth=0.05` default for optimization runs.

## 6. Benchmarks (`esbopfn/benchmarks/`)

- `db.py`: SQLite at `<PFN_HOME>/benchmarks/db.sqlite`, table `units(kind, key_hash, key_json,
  status, blob_dir, started, finished, duration_s, code_version, host)`. `Db.has(kind, key)`,
  `Db.run(kind, key, fn)` (skips when done, marks running, stores), `Db.query(kind, **filters)`.
  Blobs under `<PFN_HOME>/benchmarks/blobs/<kind>/<key_hash>/` (parquet/json/jsonl).
- Kinds (each a module in `benchmarks/kinds/`, exposing `KIND` with `enumerate(spec) -> list[key]`
  and `run(key) -> blob_dir`): `fields`, `prediction`, `prediction_ladder`, `prediction_timing`,
  `optimization`, `optimization_timing`, `generation_speed`, `landscapes`, `hardware_optimization`,
  `hardware_prediction`. Keys include the model id and checkpoint hash where a model is involved,
  problem, instance/seed, budget and the rule parameters.
- Plot groups (`plots/<group>.py` + `configs/benchmarks/<group>.yaml`): `needs(cfg) -> list[(kind, key)]`,
  `render(cfg, db, out_dir, fmt, theme)`. `pfn bench plan|run <group> [--shard i/n]`,
  `pfn plot <group> [--format pdf|png] [--theme white|black]`.
- Metrics exactly thesis §5 (S:542-681): conditioned LL, five cutoffs 10..90% snapped to the
  closest grid point before, per-trial mean then median, `ll_norm` by per-problem median, Pareto
  fronts over context sizes, stop curves (steps saved vs false-stop share), scaled simple regret
  `(J⁻ − J_best)/(J_ESRS − J_best)`, median over 20 instances, mean over problems, log10, crash
  counts, decision-time splits.
- `import_thesis.py`: readers for the old optplay run json, predplay parquet/csv, hwpred results,
  hwplay `trials.jsonl`, the timing bundle, into the DB with `provenance: thesis` keys. Thesis
  checkpoints converter into the model tree (DecPFN state_dict, LC-PFN pickled modules → state_dict,
  XDecPFN/FT-PFN ckpt → new layout; where the old architecture differs from the new one, the old
  weights are imported as `variant=thesis` and only served to the import/plot path).

## 7. Hardware (`esbopfn/hardware/`)

- `client.py`: websocket client (`reset_down`, `step(voltage)`, `center`, `close`), reply timeouts,
  reconnection. `server/`: `remote_env_server.py` with `--plant {qube,sim}`, the `qube` backend
  imports `gym_brt` if installed (BlueRiverTech driver, vendoring deferred), `sim` backend runs the
  Furuta simulator so the whole stack is testable here.
- `reset.py`: alg:app:hw-reset (A:993-1022) step by step, connection reset every tenth trial.
- `runner.py`: `pfn hw run --rig <name> --methods ... --seeds ... [--trials 180 --steps 36000]`,
  writes `trials.jsonl` per (rig, method, seed) under the DB blob dir, resumable, seeds shared across
  methods (same initial design). `configs/rigs.yaml` registry (`without_weight`, `with_weight`, each
  with a description and the J_best policy "best on this rig").
- `prediction_bench.py`: ladder n ∈ {0,1,2,5,11,22,45,90} and the four-model ablation (HR:29-95).

## 8. Cluster (`esbopfn/cluster/`)

- `configs/cluster/<name>.yaml` as in `05_proposal.md` §4 (no login-node mode).
- `pfn submit [--profile] [--kind gen|train|bench|bench_gpu] [--time --cpus --mem --gpus]
  [--array a-b] [--after id] [--name] [--dry-run] [--local] -- pfn <command ...>`. Renders
  `jobs/<ts>_<name>.sh`, logs to `logs/`, `--array` maps `SLURM_ARRAY_TASK_ID` to `--shard i/n`.
- `pfn status` (datasets, models, DB counts, `squeue` if available), `pfn doctor` (torch device,
  compilers, BLAS, TabPFN checkpoint, cpp binaries).

## 9. Plots and style (`esbopfn/plots/style.py`)

Thesis style: Charter via usetex when a TeX with `mathdesign` is available, mathtext otherwise,
font 9 pt, widths 6.3 / 3.02 / 2.02 in, dashed grid `#B0B0B0`, Okabe-Ito method colors
(DecPFN `#0072B2`, LC-PFN `#D55E00`, FT-PFN `#009E73`, XDecPFN `#CC79A7`, TabPFN `#9B4FA8`,
Segment-TabPFN `#882255`, Segment-GP/ESBO-GP `#E69F00`, ESRS `#BBBBBB`, BO `#8C564B`,
DecPFN-H `#0B3C73`, Oracle `#000000`), RWTH palette for schematics (`#00549F`, `#F6A800`,
`#CC071E`, `#57AB27`). `--theme black` inverts background/text only.

## 10. Tests (minimum per layer)

- problems: golden trajectories against the old Python simulators (generated once from the old
  code into `tests/golden/`), crash/success rules, cost formulas, prior supports, screen statistics.
- grids: the thesis table (11/7/8/15/9 strides, 64/68/63/67/67 points), token covariates.
- generators: `cpp` parity (same instance+θ rollout within tolerance, KS on J_T / crash rate),
  chunk-seed invariance to worker count, pool read/write.
- predictors: shapes, loss finite, head NLL identities, `conditioned_above`, tiny training runs
  reduce loss (`gpu`-free, CPU-sized models), old-checkpoint import round trips.
- strategies: rules on synthetic predictions, budget accounting, ESRS exactness, seeds reproduce.
- benchmarks: DB idempotence, metric formulas on toy data, importers on small fixture files.
- plots: every group renders on fixture data (pdf+png, both themes).
- cluster: profile rendering, `--dry-run` script content, `--local` execution.
- hardware: client against the `sim` server in-process.

## 11. Remaining known deviations to document in `docs/DEVIATIONS.md`

Only these are expected to remain after the build. Anything else is a bug.
- `cold_refit_growth = 0.05` in optimization runs (thesis silent). Tony may set 0.0.
- XDecPFN and FT-PFN train without gradient-norm clipping (thesis appendix states clip 1 for all four models). Decision 2026-10-28.
- Training set sizes are exact (100,000 / 1,000,000) and steps = floor(N/128).
- Specification choices where the thesis is silent: seeds, validation set sizes, ρ estimation by
  pilot sampling, iid uniform initial design, candidate set for the GP acquisition (2000 uniform).
