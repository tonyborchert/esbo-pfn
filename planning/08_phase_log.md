# 08 — Phase log and interface addenda

Append-only. Each phase agent reads this before starting and appends its own addendum at the end.

## Phase 1 (skeleton) — done, commit 2958848

`uv run pytest -q` → 51 passed. Interface decisions that bind later phases:

1. **Configs live inside the package** at `src/esbopfn/configs/` (`esbopfn.config.configs_dir()`):
   `problems.yaml`, `cluster/*.yaml`, `benchmarks/<group>.yaml`. Put `hparams/*.yaml`,
   `datasets.yaml`, `rigs.yaml` there too.
2. `esbopfn.config.paths()` → lazy subdirs `.datasets .models .hpo .checkpoints .benchmarks
   .figures .jobs .logs .bin`, `tabpfn_checkpoints`. `$PFN_HOME/bin` is where built C++ binaries
   are looked up (`furuta_gen`, `cartpole_gen`, `plants_gen`), as is `cpp/build*`.
3. `util/seeds.py`: `substream(root, *labels) -> Generator`, `derive_seed(root, *labels) -> int`
   (63-bit, for torch and C++), `torch_generator`, `seed_sequence`.
4. `core.ProblemSpec`: fields per 07 plus optional `theta_names`, property `extent`, `to_dict()`,
   `spec_hash()`. Budget helpers are methods. Furuta is named `furuta`. Channels:
   cartpole levels `[x_cart, phi_pole]` rates `[xdot_cart, phidot_pole]`, furuta levels
   `[phi_arm, phi_pend]` rates `[phidot_arm, phidot_pend]`, threetank_pi `[V2, V3, V4]`,
   boiler `[temperature]`, pt2_deadtime `[y]`. Noise bases and prior tables are NOT in the yaml yet.
5. `core.Trajectory.cut(t_obs)` caps at `crash_step`, `t_obs = 0` = unrun. `Trial` extras:
   `unrun`, `early_stopped`, `successful`, `theta`, `T`. `J_obs` is 0 when unrun, inf when crashed.
   Simulators must set `success = not crashed` on problems without a success criterion.
6. `grids.decoder_grid` returns only the N_grid−1 token steps, horizon = `spec.T`.
   `grids.tokens()` → `Tokens(t, J, covariates)` with `[ΔJ, ξ, Σξ̇]`, grid points at or after the
   crash step dropped, ξ̇ = Δξ with 0 at step 1 when no measured rates.
7. `benchmarks.db.Db`: unit fns `fn(key, blob_dir)` write into a temp dir renamed on success,
   stale `running` re-runs, failure marks `failed` and re-raises, `query(kind, status="done",
   **filters)` on top-level key fields, key hash = sha256 of canonical JSON of the key.
   `BenchmarkKind.run(key, blob_dir) -> None`. Sharding takes every n-th unit of the sorted list.
8. `pfn submit`: job `PFN_HOME` = `$PFN_HOME` at submit time else profile root
   (`${HPCWORK}/esbo-pfn`) else `./out`. Script cds to the submit directory. Leading `pfn` gets the
   profile launcher (`uv run --no-sync`). `--array a-b` appends `--shard "$((SLURM_ARRAY_TASK_ID-a))/n"`,
   so every sharded command accepts `--shard i/n`. `--kind gen` also loads `modules_cpp`.
9. `plots.style`: `thesis_style(theme, usetex=None)`, `save(fig, path, fmt)`,
   `method_color(name, theme)`, `DISPLAY_NAME` keyed by lowercase hyphenated ids
   (`decpfn-nostates`, `tabpfn`, `segment-gp`, …), `display_name()` accepts full model ids.
   Every module in `plots/` except `style` is treated as a plot group.
10. Command modules: `commands/{config,status,doctor,submit,bench,plot}.py`.

## Phase 2a (problems, prior, simulators) — done

`src/esbopfn/problems/{base,prior,registry,__init__}.py`, one subpackage per problem with `sim.py`,
`configs/problems.yaml` gained `prior:` and `noise:` sections. Golden parity with the old simulators
(`tests/golden/*.npz`, `make_golden.py`). 73 tests.

- `get_problem(name, dead_time=True) -> Problem` with `.spec`, `.prior`, `.simulate(theta, instance,
  rng, t_max=None)`, `.nominal_instance(noise=True)`. `simulate` always returns rates.
- Instance `phi` keys: boiler `heating_rate, cooling_rate`; threetank_pi `V101_opening, V106_opening,
  B101_volume`; pt2_deadtime `K, w0, D`; cartpole `cart_mass, pole_mass, pole_half_length, max_force`;
  furuta `Rm, kt, km, mr, Lr, Dr, mp, Lp, Dp`. `phi_hat` only for furuta (else `{}`). `extras`
  (furuta): `dead_time` [s], `kick` [V], kick sign drawn per trial.
- `noise_std` keys: boiler `temperature`, threetank `V4`, pt2 `y`, cartpole per channel, furuta
  `angle, velocity, voltage` (voltage = actuation noise). All channels noisy except threetank V2, V3.
- Trajectory: J float32, +inf from `crash_step` (1-based), x/xdot padded with the last value after a crash.
- Prior API: `sample_instance(problem_or_prior, rng)`, `sample_theta(spec, rng)`, `screen(problem,
  instance, rng, t_max=None)`, `single_trajectory_prior(problem, rng, max_unsuccessful_share=0.5,
  pilot=Pilot(n=2000, seed=0), rho=None) -> SingleTrajectorySampler` (`.sample() -> PriorSample(instance,
  trajectory)`, cached `rho`), `trajectory_set_prior(problem, inst, n, rng, rates=None, n_pilot=100,
  cut=True) -> TrajectorySet(trials, counts, p_cut, n_simulated)` (`cut=False` = XDecPFN pool variant).
- Thesis applied over old code: PT2 dead time exactly 10 steps, noise S·η everywhere, p̂ clipped
  normal without tie re-draw, furuta uniform dead time + kick U[0,0.6], channels per thesis tables.
- The nominal furuta plant does not pass the screen (prior acceptance ≈1%, thesis 1.5%).
- Speeds (Mac, one core): boiler 5400, threetank 1150, pt2 1600, cartpole 1050, furuta 350 rollouts/s.

## Phase 2c (predictor base, GP, Segment-GP, TabPFN baselines) — done

`predictors/{base,gp,segment_gp,tabpfn,segment_tabpfn,checkpoints}.py`, `commands/checkpoints.py`,
`docs/PREDICTORS.md`. 32 tests.

- `BarPrediction(edges, probs, p_crash)` / `GaussianPrediction(mean, var, p_crash)` with
  `log_density, log_prob_crash, cdf, survival, conditioned_above(j_obs)` (keeps p_crash),
  `truncate_below(j)` (conditions the whole mixture, p_crash grows), `sample(rng, n)` (+inf for crash),
  `mean_var()` (bars include width²/12), `quantile`. `conditioned_log_likelihood(pred, y_true, j_obs)`
  floors terms at log(1e-10).
- `Predictor` protocol + `PredictorBase` (`name, supports_unrun, context_aware, predictable_steps,
  predict, predict_many`). Registry: `get_predictor(id, problem_or_spec, **kw)`,
  `register_predictor_family(prefix, factory)` or by convention a module `esbopfn.predictors.<prefix>`
  (dashes → underscores) exposing `make_predictor(args: tuple[str,...], spec, **kw)`.
- Ids: `segment-gp`, `tabpfn/<L>`, `segment-tabpfn/<split|rate>`. Crashed queries →
  `GaussianPrediction(0, 0, p_crash=1)`, complete queries → observed J_T with zero variance.
- `gp.ExactGP` (100 Adam steps, lr 0.1, clip 100, clamps as the old port, `cold_refit_growth`
  default 0.0, `fit(..., warm=other)`), `mes_acquisition(gp, candidates, rng, n_samples=20, grid=512,
  incumbent=...)`, `sample_positive` (exact inverse-CDF truncated normal).
- TabPFN: v3 regressor `tabpfn-v3-regressor-v3_default.ckpt` loaded by absolute path from
  `$PFN_HOME/checkpoints/tabpfn/` (installed here from the timing bundle via `pfn checkpoints
  download tabpfn --from FILE`), `n_estimators=8`, full bar output → `BarPrediction`.
  Download through the package needs a Prior Labs token (`TABPFN_TOKEN`).
- `pfn doctor` should use `predictors.checkpoints.tabpfn_status` (todo for a later phase).

## Phase 2b (C++ generators, pools, `pfn data`) — done

`cpp/` (CMake, `-DBLAS=accelerate|openblas|mkl`, binaries `furuta_gen cartpole_gen plants_gen`),
`generators/{pools,datasets,binaries,speed,verify}.py`, `commands/data.py`, `configs/datasets.yaml`,
`docs/DATA.md`. 14 unit + 47 cpp tests. Build: `pfn data build` (→ `$PFN_HOME/bin`).

- Parity: identical crash steps and J_T rel. error ≤ 6e-8 vs the old Python, exact agreement with
  `esbopfn.problems` on deterministic cases, distribution tests pass.
- Pool row layout (single-trajectory): `theta[d] | phi | phi_hat (furuta) | noise_std | extras
  (furuta dead_time, kick) | crash_step (1-based, 0 = none) | success | J[S] | x[S,C] | xdot[S,C_rate]`.
  `Pool.open(path)`, `Pool.read(i) -> Trajectory`, `Pool.instance(i) -> Instance`.
- **XDecPFN set pools are stored on the decoder grid** (DecPFN covariates per grid point, xdot summed
  per interval), `instance` index column, `instances.json` (parameters, screen stats, counts), 128
  unrun label trials per instance in `labels/`. `--resolution full` exists. The XDecPFN loader does the
  per-example subsampling, cutting and unrun-query selection (M:1122-1132) itself.
- FT-PFN context shards: `shards/chunk_<k>/` raw little-endian arrays + `shard.json`: per example
  `example_ptr, example_n, example_meta`, per trial `trial_theta, trial_len, trial_status (0 complete,
  1 truncated, 2 crashed), trial_target (inf on crash), trial_tok_t (int16, 9 slots, 0 = pad),
  trial_tok_J (inf = crash)`, per example 16 unrun `unrun_theta, unrun_target`. Tokens = 8-point curve
  grid steps ≤ T_i plus T_i when truncated.
- Chunk seeds: `derive_seed(seed, k)` ported bit-exactly to C++. Output independent of worker count.
- Spec choices: ρ (single-trajectory rejection) from a 50,000-draw pilot (seed 20261028), set-prior
  rates from the screen's first repeats, Sobol θ design in the C++ screen (Python screen is iid),
  `forced` flag when rejection sampling cannot fill a set.
- Throughput (Mac, one core): boiler 84k, threetank 34k, pt2 37k, cartpole 37k, furuta 470 traj/s.
- `pfn data gen --for train <model>` maps lcpfn, lcpfn-last, decpfn, decpfn-nostates,
  decpfn-nodeadtime, xdecpfn, ftpfn to datasets. `fields/*` are placeholders for the benchmark phase.

## Phase 3 shared contract (training core) — binding for T1 (owner) and T2 (consumer)

`src/esbopfn/training/common.py` (written by T1 first thing):
```python
@dataclass
class TrainConfig:
    lr: float; weight_decay: float = 0.0; dropout: float = 0.0
    batch_size: int = 128; steps: int = 0            # steps = total optimizer steps
    warmup_frac: float = 0.25                        # linear warmup from 0 to lr, then cosine to 0, per step
    clip_grad_norm: float | None = 1.0               # None = no clipping (XDecPFN, FT-PFN)
    bf16: bool = True; seed: int = 0
    time_limit_s: float | None = None                # HPO wall-clock cap
    checkpoint_every: int = 500; eval_every: int = 500; log_every: int = 50
    device: str = "auto"                             # cuda > mps > cpu

@dataclass
class FitResult: steps_done: int; train_time_s: float; final_metrics: dict; stopped_by: str

def fit(model: torch.nn.Module, batches: Iterator[Any],
        loss_fn: Callable[[torch.nn.Module, Any], torch.Tensor],
        cfg: TrainConfig, out_dir: Path,
        evaluate: Callable[[torch.nn.Module], dict[str, float]] | None = None) -> FitResult
# AdamW, schedule as above, bf16 autocast on cuda, clipping, seeded init via torch_generator,
# writes out_dir/ckpt_last.pt every checkpoint_every steps and resumes from it, metrics.json with
# the loss curve and evaluate() results, logging. `batches` is restarted by the caller if exhausted.
```
`src/esbopfn/training/registry.py` (T1): `model_dir(model_id) -> Path` under `$PFN_HOME/models/`,
`resolve(model_id) -> Path` raising `ModelNotTrained(model_id, hint)`, `save_model(out_dir, state_dict,
model_cfg: dict, calib: dict, train_cfg: dict, metrics: dict)`, `load_model(path) -> ModelFiles`
(`state_dict, model_cfg, calib, train_cfg, metrics`), `list_models()`.
Model dir contents: `model.pt` (state_dict + model_cfg), `config.yaml` (resolved hparams, dataset id,
seed, git hash, provenance), `calib.json` (bin edges, z-score stats), `metrics.json`.
Model ids: `<model>/<problem>/<variant>/<size>[/<run>]`, e.g. `decpfn/cartpole/binned/1M`,
`decpfn-nostates/cartpole/binned/1M`, `lcpfn-last/furuta/binned/100k`, `xdecpfn/cartpole/default/default`,
`ftpfn/boiler/default/default` (variant/size = `default` where there is only one).
Hyperparameter files `src/esbopfn/configs/hparams/<model>.yaml`, one row per (size, head).
