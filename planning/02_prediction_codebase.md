# 02 — Audit of `impl/prediction` for the clean rebuild

Scope: `/Users/tonyborchert/Documents/Uni/bachelorarbeit/impl/prediction` (abbreviated `P/`), read-only,
2026-10-28. No project Python was run. `D/` = `../dynformer`. Thesis = `writing/thesis/sections/*.tex`.
File:line references are to the current working tree.

**Five headline facts the rebuild depends on**
1. **Thesis numbers come mostly from `D/`, not from `P/benchmark`.** The prediction benchmark is
   `D/predplay`, the simulated optimization benchmark is `D/optplay`, the hardware prediction benchmark is
   `D/experiments/hwpred/hwpred_ladder.py`, and the thesis figures are `D/thesis_figures`. `P/` supplies
   the simulators, priors, pools, predictors, optimizers and the rig driver, all imported by `D/` through `sys.path`.
2. **No checkpoint, pool, grid `config.yaml` or `trials.csv` exists locally.** `find impl -name '*.pt'` returns 0.
   The `models/` tree (`MODELS`, default `P/models`) and the pools (`$HPCWORK/datasets`) live only on the HPC.
   The hyperparameter tables are reconstructible only from `writing/thesis/data/{PROCEDURE,RESULTS}.md` and the figure sidecars.
3. **`results_facts.md` is stale on LC-PFN.** Since commit `933a3e2` (2026-09-29), every thesis LC-PFN
   series is `lcpfn_zscore/epochs{10,100}/head*`, i.e. the `<head>_zscore` leaves, not `lcpfn/...`.
4. **Thesis "Furuta" in simulation = dataset `furuta/lqr-new-realistic/rigreset(1,False)`.** That is a uniform
   dead-time sampler with kick ~ U(0, 0.6) V (`D/optplay/optplay/problems.py:159-179`).
   **The rig DecPFN is a different checkpoint**, trained on `furuta/lqr-new-realistic/rigreset` (*stratified* dead time,
   `hwplay/field.py:172-184`).
5. **Nothing is seeded end-to-end.**
   - Python pool generation is time-seeded (`manager/dataloading.py:25-30`).
   - Model init is unseeded.
   - Reservation seeds use `hash(tag)` (`manager/register.py:233`), which is PYTHONHASHSEED-randomised.
   - C++ pools are deterministic given `(seed, workers, existing rows)`, but HEAD does not reproduce the binary that wrote them (`tools/furuta_cpp/CLAUDE.md:149-152`).

---

## 1. MODULE MAP

Line counts exclude notebooks/old. Verdicts are for "reproduce the thesis exactly".

| dir | what | size | verdict |
|---|---|---|---|
| `model/` | `Trial`/`RawTrial` data model and `get_formatted` (`system.py`), `Experiment` builder with rejection guards (`experiment.py`), `Predictor`/`Prediction` (`predictor.py`), `Optimizer` ABC with Prediction/Performance logs (`optimizer.py`), `TrialFolder` (`recording.py`) | 1.2k py | **KEEP** (drop `transform.py` once the last `dataset.py` user is gone) |
| `util/` | `distributions.py` (`Uniform`, `LogUniform`, `PosNormal`=truncated normal, `Composite`, `Resampled`), `matlab_gp.py` (GPML-layout GP + `GP_COLD_REFIT_GROWTH`), `optim/` (typed Ax-1.3/Optuna sweep library used for every training sweep), `math.py`, `stopwatch.py`, `mpl_styles.py`, `notebook.py` | 4.2k | **KEEP** `distributions`, `matlab_gp`, `optim`, `math`, `stopwatch`. Drop `notebook`. `mpl_styles` only if the plotters are kept |
| `experiments/` | 5 plant packages + `priors.py` + furuta rig tooling | 13k py, 38 MB | **KEEP-PARTIAL**, see section 2. Keep `priors.py`, the `system.py`/`objective.py`/`__init__.py` of each package with only the `universal` / `rigreset` entries, plus `furuta/{system,controller,gym,objective,realistic,tick_reset}.py`. Keep for hardware: `furuta/{hardware,fast_hardware,hardware_benchmark,reset_qube,hwlog}.py`. **DROP** ~20 furuta probe/bench/plot scripts (`pid_*`, `plot_*`, `reset_replay*`, `start_condition_bench`, `setpoint_step_bench`, `measure_limits`, `inspect_reset`, `check_repeatability`, `board_probe`, `render3d`, `record_real_experiments`, `run_hardware_benchmark`, `fast_bench`), `real_experiments/` (28 MB, not thesis), and every `example/robust/general/distorted` variant |
| `manager/` | `load.py` CLI (pregen/train/--slurm), `register.py` (reservation, model-id resolution, `@train`/`@inference`), `pool.py`, `dataloading.py`, `storage.py`, `base.slurm` | 0.9k | **KEEP**. Simplify, and make paths and accounts configurable |
| `predictors/decpfn` | DecPFN (`config.py`, `architecture.py`, `train.py`, `optim.py`, `val.py`, `infer.py`, `record.py`) | 1.3k | **KEEP** |
| `predictors/lcpfn` | LC-PFN. `custom.py` holds config, train, validate and predictor (1222 lines). Model code is a vendored copy of the released LC-PFN (`transformer.py`, `layer.py`, `encoders.py`, `bar_distribution.py`, `utils.py`, `gaussian_curve.py`) | 3.5k | **KEEP** `custom`, `config`, `transformer`, `layer`, `encoders`, `bar_distribution`, `utils`, `gaussian_curve`. **DROP** `domhan_prior.py`, `curves.py`, `train.py`, `train_lcpfn.py`, `model.py`, `priors/`, `decoders.py`, `initializers.py`, `positional_encodings.py`, `version.py` (none is imported by `custom.py`; check with grep). **Caveat:** LC-PFN checkpoints are pickled whole modules (`torch.save(model)`, `custom.py:~995`), so the class import paths `predictors.lcpfn.transformer` etc. must survive or a converter is needed |
| `predictors/tpfn` | Trajectory-TabPFN (`raw` method used; `ls`/`diff` not in thesis) | 0.5k | **KEEP** (`raw` only) |
| `predictors/stengertpfn` | Segment-TabPFN (`split`, `rate` used; `minimal`/`anchored` not) | 0.5k | **KEEP** (`split`, `rate`) |
| `predictors/stenger` | Segment-GP = ESBO-GP section GPs | 0.3k | **KEEP** |
| `predictors/lsgp` | global time-axis GP | 0.25k | **DROP** (not in the thesis, no optimizer uses it) |
| `optimizers/` | BO loop + every early-stopping variant | 3.4k | **KEEP-PARTIAL**, see section 7 |
| `benchmark/` | `Sampler`, `PredictorBenchmark`, `OptimizationExecution`, `OptimizerBenchmark`, plotters, cartpole Pareto scripts | 6.3k | **KEEP-PARTIAL**: only what `D/` imports (section 7). The `aggregate_cartpole_pareto.py`/`plot_cartpole_pareto_hparams.py` scripts are robust(2)-era, **DROP** |
| `tools/furuta_cpp`, `tools/cartpole_cpp`, `tools/plants_cpp` | C++ prior samplers (section 3) | 1.2k py + C++, 11 MB | **KEEP** (drop the checked-in ELF `furuta_gen_phi` and `cartpole_gen_s11` side builds; rebuild from source) |
| `training/lcpfn` | legacy copy of the original LC-PFN repo (Domhan prior, sweep.sh, results PNGs) | 4.8k | **DROP** (self-contained, imported by nothing outside itself) |
| `hwplay/` | real-rig optimizer field (config, field, run, aggregate, figures, selfcheck) + run trees | 3.5k py, 978 MB | **KEEP-PARTIAL**: code plus the two thesis run trees (section 8). DROP `prediction_benchmark.py`, `prediction_scatter.py` and the 6 obsolete run trees |
| `furuta_rigreset_models/` | MANIFEST of 12 rigreset DecPFN checkpoints (`.pt` not present locally) | 20 KB | **KEEP** manifest. Fetch 5 `.pt` (section 8) |
| `c1r2_furuta_lnJ/` | XDecPFN rig calibration json + note | 40 KB | **KEEP** (belongs to the XDecPFN side) |
| `scripts/` | ~106 probe/report/slurm scripts, mostly prior design (`prior_v2..v5`, `*_probe.py`) | 11.5k | **DROP** except `export_rigreset_models.py`, `furuta_rigreset_rates.py` (cached success rates) and `compile_hw_trajectories.py`/`hwplay_to_npz.py` if a thesis figure needs them. `furuta_hardware_predictions*` is old robust(2) work and is DROP |
| `handoff_matched_live_prediction/` | 2026-08 research handoff (results CSVs, PROGRESS/RANKING) | 7k py, 60 MB | **DROP** (archive only) |
| `notebooks/` | demos, `big_benchmark*.py` | 191 MB | **DROP** |
| `ESBO-main/` | reference MATLAB | — | not code. Cite it, optionally vendor as reference |
| `OKAY`, `OPTIMIZER_CONDITIONING.md`, `PYTHONPATH=.:util/` | junk / notes | — | **DROP** (`OPTIMIZER_CONDITIONING.md` is useful reading for section 7) |

**Entry points per kept capability**
- Pool generation:
  - Python: `PYTHONPATH=.:util python manager/load.py pregen <dataset> <count> [--num_workers W] [--slurm -A acct]` (`manager/load.py:235-277`).
  - C++: `tools/furuta_cpp/furuta_gen gen --dataset 'furuta/lqr-new-realistic/rigreset(1,False)' --count N --workers W --seed S` and `tools/cartpole_cpp/cartpole_gen gen --dataset cartpole/universal ...`.
  - plants_cpp has **no** `gen` (section 3).
- Training:
  - `manager/load.py train decpfn <dataset> --max_budget N [--name grid/100] [--pre-gen] [--slurm]` → `predictors/decpfn/optim.py:90-148` (`run`).
  - `manager/load.py train lcpfn <dataset> --max_budget N --name ...` → `predictors/lcpfn/custom.py:1006` (`run`).
  - The optimizer folder must already contain `config.yaml` (the search space and fixed block, `util/optim` spec).
- Inference: `manager.load.load_predictor("decpfn/<abs path to model.pt>")`, likewise `lcpfn/...`, `"stenger/<dataset>"`, `"tpfn/raw/<dataset>"` (kwargs `version="3", sample_size=L`), `"stengertpfn/{split,rate}/<dataset>"` (kwarg `version="3"`). This is exactly what `D/predplay/predplay/registry.py:243-283` does.
- Optimization: the classes in `optimizers/` driven by `D/optplay` (sim) or `hwplay/run.py` (rig).

---

## 2. PROBLEMS

Every experiment is a package `experiments/<name>/{system.py, objective.py, __init__.py}` exposing an
`experiments` dict, auto-discovered by `experiments/__init__.py`, resolved with `resolve_dataset("name/variant(args)")`.

### 2.1 The `universal` recipe (`experiments/priors.py`)

- **Scale params:** `LogUniform(nominal/s_lo, nominal*s_hi)`.
  - Spans are `PHI_SPAN` (`priors.py:~115-160`); the default span is 3.0 (`universal_phi`, `priors.py:~308`).
  - `k` exponentiates the spans (`_span`, the calibrated `k=1`).
  - The nominal phi is each package's `DEFAULT_PHI` / `default_phi`.
- **Bounded params:** `BoundedMix` → `Uniform` inside `PHI_DOMAIN` (threetank only).
- **Tied params:** `TiedTo` → `km = kt · LogUniform(1/1.5, 1.5)` (furuta, `PHI_TIE`).
- **Noise:** `SharedLogUniform`, `noise_i = base_i · S · J_i`.
  - Multi-channel: `S ~ LU(1.5/4, 4/1.5)`, `J_i ~ LU(1/1.5, 1.5)`.
  - **Single-channel (boiler, pt2, threetank): jitter dropped and `S ~ LU(1/4, 4)`** (`universal_noise`, last function).
  - `NOISE_BASE`: cartpole `[0.004, 0.02, 0.002, 0.02]`, pt2 `4e-5`, threetank `1.7e-4`, boiler `0.03`, furuta `[0.002 angles, 0.07 velocities, 0.1 V]`. Furuta's 3 bases cover 5 channels, so both angles share one jitter draw.
- **Estimate (furuta only):** `universal_phi_hat` → `PosNormal(phi, phi/10)` (a **rejection-truncated** normal, `util/distributions.py:66-95`), then the km/kt tie is **re-drawn on the estimate**.
  - realistic: a 10th entry fixed at `0` (`realistic.py:313-326`).
- **Success guard:** `Experiment.successful_share(0.5)` / `failed_share(0.5)`.
  - Rejection probability `(s-q)/(s(1-q))` (`model/experiment.py:160-172`) uses a **cached** natural rate (`set_cached_*`).
  - It equals the thesis formula `max(0, 1-(1-ρ)/ρ)`.
  - The draw uses Python's global `random.random()` (`experiment.py:205-207`).
- **Screening** (the degeneracy screen: 100 Sobol thetas, ≥25 succeed 10×) is **not** part of the Python `Experiment`.
  - It lives in the C++ `ctx`/`probe` and in `D/optplay/screen.py` for the benchmark instances.
  - Single-trajectory pools are unscreened, as the thesis says.

### 2.2 Per problem

All `T` are logged steps. "states" = the `y`/`Y` blocks DecPFN receives (`Y` = level, `y` = per-step diff, interval-summed under `add_diffed_up`).

| | boiler | threetank_pi | pt2_deadtime | cartpole | furuta (thesis) |
|---|---|---|---|---|---|
| dataset | `boiler/universal` | `threetank_pi/universal` | `pt2_deadtime/universal` | `cartpole/universal` | `furuta/lqr-new-realistic/rigreset(1,False)` (sim/predplay/optplay), `.../rigreset` = stratified (rig DecPFN) |
| System / Objective | `BoilerBangBangSystem` / `BoilerObjective` | `ThreeTankPISystem` / `ThreeTankObjective` | `PT2DeadTimePIDSystem` / `PT2Objective` | `CartPolePIDMujocoSystem` (a gymnasium port despite the name, Euler τ=0.02, `cartpole/gym.py:135-174`) / `CartPoleCost(diag(10,1,1,1), tanh on pos)` | `FurutaRealisticSystem`+`FurutaTickResetSystem` / `FurutaHoldCost(thr 5°, w 20)` (`objective.py:80-134`), success = `FurutaHeldCost` (mean \|α\| final 2 s < 5°) |
| T, dt | 701, log 2 s (sim 1 s) | 476, log 0.2 s (dt 0.1) | 501, log 0.1 s (dt 0.01) | 1000, 0.02 s | 600, 120 Hz |
| θ box | TurnOnDelay [5,500] | Kp [0,3], Ki [0,0.1] | log10 Kp,Ki,Kd ∈ [-1,1]³ | `THETA_BOX["cartpole"]` [0.3,30,10,150]..[3,300,100,1500] | `THETA_BOX["furuta/lqr"]` [40,.7,11.5,4.5,.5,.5,.5,1.5]..[90,1.2,12.5,5.5,1.5,1.5,1.5,3.5] |
| crash | none | none | y−w > 2 (`system.py:46`) | \|x\|>3.5 or \|φ\|>30° (`gym.py:119-182`) | \|arm\|>90°, non-finite, undesignable LQR |
| guard | none | none | `failed_share(.5)`+`successful_share(.5)`, cached .5093/.49005 | `failed_share(.5)`, cached .61630 | `successful_share(.5)`, cached .02059 (Py) / .02225 (C++ uniform) |
| states Y (y = diff) | **[meas temp, burner cmd u]** | [V2, V3, meas V4] | [meas y] | **[pos rel. goal, pole angle, action]**, y=[cart vel, pole rate, Δaction] | **[θarm, α, θ̇, α̇, target≡0]**, y = per-step diffs of all 5 |
| C++ twin | `plant_gen` (ctx/raw/stats only) | `plant_gen` | `plant_gen` | `cartpole_gen` (gen+ctx) | `furuta_gen` (gen; ctx per-instance) |

**Known mismatches with the thesis text (setup.tex, appendix tables)**

- **P1. States exposed to DecPFN differ from `tab:app:problems`.** Code builds features with `get_formatted(("y","Y","j","J","failed"))` (`predictors/decpfn/train.py:86-97`).
  - Boiler also feeds the burner command `u`.
  - Cartpole feeds the bang-bang action and its diff, and the position *relative to the goal*.
  - Furuta feeds a constant `target` channel (13 input columns, `furuta_rigreset_models/MANIFEST.md`), and treats velocities as levels whose interval *increments* are the "rates". The thesis splits them as ξ = angles and ξ̇ = measured velocities.
- **P2. Noise jitter.** The thesis formula (`setup.tex:337-348`) applies `S·J_i` to every problem. Code drops `J_i` for the three 1-channel problems and uses `S ~ LU(1/4, 4)`. The marginal support base/4..4·base is the same.
- **P3. φ̂ estimate.** The thesis says `max{0, N(p, (p/10)²)}` (clip). Code uses a rejection-truncated normal, plus the km/kt tie re-applied on φ̂. The thesis does not mention the tie on φ̂.
- **P4. Furuta dead-time sampler.**
  - The thesis (`tab:app:prior-phi`) says "Bounded" uniform [8.33, 25] ms. That holds for the sim models (`rigreset(1,False)`).
  - The rig DecPFN (`hwplay/field.py:172-184`) uses `StratifiedDeadTime` (`realistic.py:261-301`), which reweights the *draws* ∝ 1/held-rate.
  - The rig ablation's "dead-time + states" arm uses the **uniform** checkpoint, so "the original" in `hw_results.tex:83-85` is not the deployed model.
- **P5. "All priors are implemented in C++"** (`appendix.tex:~876`). `plants_cpp` has no pool writer. Boiler/threetank/pt2 single-trajectory pools come from Python `manager/load.py pregen`. Only their trajectory-set (`ctx`) streams are C++.
- **P6. Cartpole naming.** The system class is called "Mujoco" but is a gymnasium-style Euler integrator (matches the thesis). Its noise uses the unseeded global `np.random.normal` (`cartpole/system.py:69`).

---

## 3. C++ GENERATORS (`tools/*_cpp`)

**Common**
- `g++ -O3 -march=x86-64-v3 -std=c++17 -pthread -static-libstdc++ -static-libgcc`, no fast-math (Makefiles :14-28).
- One `std::thread` per worker, no OpenMP.
- RNG `mt19937_64` + splitmix64 (`furuta_cpp/dataset.hpp:46-75`). It is **not bit-identical** to Python, which only claims the same distributions.
- Output root: `$DATA`, else `$HPCWORK/datasets`, else `/tmp/datasets` (`furuta main.cpp:69-73`), identical to `manager/storage.py:8-9`.

**furuta_cpp (`furuta_gen`)**
- **Build:** `module load GCC/13.3.0 imkl/2024.2.0 && make -C tools/furuta_cpp`. Needs MKL (sequential LAPACK `dgges`/`dtgsen` for the scipy-identical DARE port). `make` fails if `MKLROOT` is unset.
- **Subcommands:** `gen|stats|raw|ctx|gains|traj|probe|slice` (`main.cpp:11-23`).
- **Datasets:** `furuta/lqr-new[-realistic]/rigreset(...)`, with Python-style positional args (`dataset.hpp:430-585`).
  - Realistic defaults to `stratified=True`, matching Python.
- **Per-plant extensions:** dead time `U(1,3)` samples, stored in seconds as the 10th phi entry, or the stratified sampler with strata held-rates 0.0361/0.0266/0.0167/0.0078 (`dataset.hpp:221-248, 665-681`).
  - Kick `U(kick_lo, kick_hi)` is drawn per plant (not stored in the record); its sign is per trial. 30 ms pulse, residual std 0.01, `RESET_MAX_STATE` fallback (`dataset.hpp:266-285`).
- **Pool on disk** (`<DATA>/<dataset>/`):
  - `pool.npy_raw` is raw float32 `(N, R)` with no header.
  - `pool.spec.json` is `{"record_size":R,"T":T}`, plus a `pool.finished` marker.
  - Records follow the Python `Trial` field order: `theta[D] | phi[9(+1)] | failed[T] | x[T] | X[T] | y[T×5] | Y[T×5] | j[T] | J[T]`, zero-padded. `failed` is 1 from the crash step on.
  - R = 9018 floats for realistic (~36 KB per row). The thesis pool of 1,040,016 rows is ≈ 37.5 GB.
  - `gen` grows an existing compatible pool. `--fresh` discards it.
- **Throughput:**
  - ~1500 samples/s on 48 workers (general(10)).
  - realistic needs ~22.6 rollouts per stored sample, universal ~9.
  - ctx runs ~4900 rollouts/s/core vs ~54 in Python.
  - On realistic/rigreset 1 candidate rig in 69 passes the screen (≈ thesis' 0.015 system acceptance) (`main.cpp:572-576`).
- **Reproducibility:** worker seed = `splitmix64(seed ^ splitmix64(c*0x9E3779B9 + have*…))` (`main.cpp:288-290`).
  - Deterministic given (seed, workers, rows already present), **not** invariant to the worker count.
  - ctx seeds per instance index. The Sobol screen design is `sobol.hpp`.
  - `CLAUDE.md:149-152`: the binary that wrote the Aug/Sep pools does not reproduce from HEAD.
- **Parity:** `tools/furuta_cpp/validate.py` (+ `validate.slurm`). The checks:
  - gains tol 1e-7
  - traj: ≥75% same t, median rel J < 1e-4
  - raw: KS p > 0.001 on J_T, t, t_d
  - prior: box/support/KS
  - pool: y = diff(Y), cumsum j ≈ J, success ≥ 0.35
  - Measured on lqr-new/rigreset: crash 0.6985 (Python) vs 0.6937 (C++), success 0.0587 vs 0.0542.

**cartpole_cpp (`cartpole_gen`)**
- No MKL. Subcommands `gen|stats|raw|traj|ctx|probe|scan|slice`.
- `cartpole/universal[(k)]` with `failed_share(.5)` and cached 0.61630.
- Record R = 11008 floats (44 KB; the 1M pool ≈ 44 GB, write-bound).
- ~900 stored samples/s/core.
- ctx: screen ON by default (100 θ × 10 repeats, 0.25), instance = f(seed, i).
- Parity: `validate.py` (traj fails its own strict bar 101/120 on universal; the docs blame the bar). Raw crash 0.6132 (Python) vs 0.6130 (C++).
- `D/` pins a side build `cartpole_gen_s11` that is not in the Makefile.

**plants_cpp (`plant_gen`, boiler/threetank/pt2)**
- Needs `CXX=g++` (the login shell exports icpx).
- Subcommands `ctx|raw|stats|traj|probe|slice`. **No `gen`.** The ctx screen defaults OFF.
- Parity lives in `D/experiments/f6/*` and `D/experiments/boiler_port/`: noise-off bit-identical, noise-on KS clean at n = 24k.

**Generation-speed table** (`tab:app:prior-cost`): **no script in `P/` or `D/` produces it.**
- Closest candidates: `D/experiments/boiler_port/BP_throughput.sh` (`plant_gen_ctx2 stats --workers 1`), `D/experiments/f17/f17_throughput.sh` (cartpole ctx), and `manager/load.py:97-109` `_estimate_gen_speed`.
- The rebuild must add a `bench_prior_cost` script, or mark the table as historical.

---

## 4. DATA POOLS / MANAGER

- **CLI** (`manager/load.py:235-277`):
  - `pregen <dataset> <count>`, `pregen-assess <model> <dataset> [--generate]`
  - `train <model> <dataset> [--pre-gen | --pre-gen-mem | --online-gen]`, `show_args`
  - Common flags: `--num_workers`, `--slurm -A <acct> --time --job-name-suff`.
  - Unknown args pass to the predictor's `@train` (`--max_budget`, `--name`, `--onfly_subsample`, `--record_gifs`, `--rerun_failed`).
- **`--slurm`** (`load.py:205-232`):
  - partition `c23g` + `--gres=gpu:1` for train, `c23ms` otherwise
  - cpus = workers + 1, mem 32G
  - logs to `logs/<name>.{out,err}`
  - `load.py` still resolves the dataset on the login node, so it imports numpy.
- **Paths** (`manager/storage.py:5-9`): `MODELS` env (default `./models`), `DATA` env (default `$HPCWORK/datasets`, else `/tmp/datasets`). Pool = `DATA/<dataset string>/pool.npy_raw`, so the **dataset string with its args is the directory name**.
- **Python reading** (`pool.py:22-60`, `dataloading.py:49-139`):
  - `TrialSpec` learns the record layout from one fresh padded Python sample (`spec_of`, `register.py:159-162`). The C++ layout therefore must match it exactly.
  - Sidecar JSON equality check.
  - `np.memmap(float32)` per worker; `shuffle=False`, `prefetch_factor=4`.
  - On-the-fly subsampling to the trial's grid inside the worker (`--onfly_subsample 1`, the default).
- **Reservation** (`register.py:175-218`): `reserve(tag, size, transform, share)` hands out fixed contiguous row ranges **in call order**.
  - DecPFN order: training = epochs × 10,240, then validation 3000, then gif 10 (`decpfn/optim.py:20-43`).
  - LC-PFN order: validation 1024 **first**, then training (`lcpfn/custom.py:108, 920-944`).
  - `manager/CLAUDE.md:58` ("val first") is wrong for DecPFN.
  - There is no separate eval pool. Validation is a reserved slice of the same pool.
- **Training sizes:** `EPOCH_SAMPLES = 2**10*10 = 10240` (`decpfn/config.py:16`; lcpfn `SAMPLES_PER_EPOCH`, `custom.py:25`).
  - The thesis "100k" = 10 epochs = **102,400** examples; "1M" = 100 epochs = **1,024,000** examples.
  - Each example is fresh, no recycling.
  - Pools: furuta realistic 1,040,016 rows (MANIFEST; ≈ `--count 1040000` on 48 workers, inferred). The LC-PFN furuta zscore pool was grown to 1,025,024 (`lcpfn_decpfn_heads_benchmark.md:14`).
  - The cartpole/universal pool (45 GB) was **deleted and regenerated 2026-09-20 with a time-seeded generator**, so the gaussian grids and the zscore re-trains for boiler/pt2/threetank saw different draws than the originals (`gridsearch_panels.md`, Data section).
- **Thesis model ids → files** (`D/predplay/predplay/registry.py:243-283`):
  - Path: `P/models/<family>/<dataset>/transfer/<epochs>/<head>[_nostates][_zscore]/0/model.pt`.
  - `decpfn/epochs100/headbinned` → `models/decpfn/<ds>/transfer/100/binned/0/model.pt`
  - `.../states0` → `.../100/binned_nostates/0/...`
  - **LC-PFN (thesis)** → `models/lcpfn/<ds>/transfer/100/binned_zscore/0/model.pt`
  - Each `transfer` leaf is the strict-min `val_accuracies[(20,99)]` cell of the cartpole `grid/` sweep, re-trained per problem (each leaf has a `SOURCE.txt`).
- **Eval sets for the benchmarks:** the queries are frozen in `D/predplay` (taken from ESBO-Oracle runs on screened instances) and the 20 instances per problem in `D/optplay`. Neither lives in `P/`.

---

## 5. PREDICTORS

**Interface** (`model/predictor.py`): `Predictor.predict(partial: Trial, actual: list[Trial], low_fid, predict_only_last) -> Prediction` plus `predictable_timesteps(partial)`.
- `Prediction` is `HistogramPrediction` (bar) or `GaussianPrediction`.
- Value statistics are conditional on no failure. `failure_prob` is a separate channel.
- Batch interface for the segment predictors: `predict_finals(partials, actual) -> [(mean, std, failure)]`.

### DecPFN (`predictors/decpfn`)
- **Architecture** (`architecture.py:219-287`): `nn.TransformerEncoder` with `norm_first=True` (pre-LN) and a causal mask, then a final `LayerNorm`.
  - Input = `Linear(n_features+1 → d)` over `[normed features, t]`, where t = `linspace(-√3, √3, L-1)`, a standardized-uniform token index.
  - `d_ff = hidden_mult·d`. Activation is chosen (gelu in all searches).
- **Input norm:** `linear` = z-score with per-feature mean/std from a 10k-row batch of non-failed positions (`train.py:146-170`). `binned`/`gaussian` are quantile-CDF norms.
- **Features:** `(y, Y, j, J)` + `failed` (pad channel). With `extra_inputs=False` only `(j, J)` (DecPFN-noStates, `train.py:375-376`; infer `:24-26`). `DECPFN_DROP_Y` env is not used in the thesis.
- **Grid:** `np.arange(0, T-1, res)` + `[T-1]` (`train.py:95,139`; infer `:17-19`). The model reads the first L-1 points and predicts point L.
- **Heads** (`architecture.py:112-216`):
  - `BinnedHead`: quantile edges over J_T, 1000 bins + failure bucket 0, out-of-range targets clamped into the edge bins.
  - `LinearBinnedHead`: equal width.
  - `GaussianHead`: z-scored target with the J_T mean/std, sigmoid failure logit, target clamped to the binned support.
- **Checkpoint:** a state_dict + config dict, all CPU, portable (`save`/`load`, `:297-349`).

### LC-PFN (`predictors/lcpfn/custom.py`)
- **Architecture:** the vendored released `TransformerModel`, post-LN (`pre_norm=False`), GELU, decoder `Linear-GELU-Linear` (`transformer.py:64-68`), with the PFN zero-init of `linear2`/`out_proj`.
  - x-encoder = grid index / (L-1), uniform-standardized.
- **y path:** cost is normalized to [0.02, 1] by `cap` = max non-failed cost of 10k calibration curves (`calibrate_spec`, `:204-219`); failure = 0.
  - `input_norm` ∈ `linear` (equal-width CDF ≈ min-max) / `binned` / **`zscore`** (calibration mean/std, the thesis models).
- **Grid:** `subsample_grid = linspace(0, T-1, ceil((T-1)/res)+1)` (`:83-87`), including index 0.
- **Heads:** `FailureBarDistribution` (failure atom bucket 0 + 1000 value bins, linear or quantile edges over **all** curve values) or `GaussianCurveDistribution` (`gaussian_curve.py`, added 2026-09-15).
- **Checkpoint:** `torch.save(model)` pickles the whole module, with metadata attributes `norm_spec`, `input_norm_edges`, `output_type`, `mask_crashed`, `resolution`, `seq_len`.
- **"LC-PFN trained only on the last input/target": does not exist.**
  - Grep over `P/` and `D/` (`only_last`, `last_only`, `final_only`, `target_last`, `LCPFN_LAST`, git log) finds nothing.
  - LC-PFN always trains on the whole-curve loss over the `T-k` future grid points. Only *validation* scores the last point (`validate`, `:713-`). `predict_only_last=True` merely slices the prediction at inference.
  - Building it would be new work: a target mask restricted to index L-1 in `train_one`'s `criterion(...)` call (`:625-628`).

### Trajectory-TabPFN (`predictors/tpfn`)
- Thesis id `tabpfn/resolution{L}/typeraw` = `load_predictor("tpfn/raw/<ds>", version="3", sample_size=L)` (`registry.py:271-275`).
- Rows `[t, θ…, fidelity_flag]`, target = raw cumulative J. The grid is `linspace(0, T-1, L)` including index 0 (`_grid`, `:79-85`).
- TabPFN regressor, `n_estimators=8`, `inference_precision="auto"`, device mps > cuda > cpu (`:70-76, 186-196`).
- **Dependency:** `tabpfn==8.1.0` (uv.lock; pyproject `>=8.0.5`), `tabpfn-client 0.3.3` (optional hosted mode).
  - Checkpoint `tabpfn-v3-regressor-v3_default.ckpt` is expected at `predictors/tpfn/models/` (`MODELS_DIR = Path(__file__).parent/"models"`, `:34-41`).
  - That directory is **not** present locally. It must be downloaded (TabPFN's HF release) and the path made configurable.
  - The package `CLAUDE.md` still says "TabPFN-2.5". The thesis uses version "3".

### Segment-TabPFN (`predictors/stengertpfn`)
- `stengertpfn/{split,rate}/<ds>`, `version="3"`, default `approximation="gaussian_fit"` (moment-match each section and sum, means clamped ≥0), `n_estimators=8`.
- `rate` uses `x=[θ, section index]`, `y = cost/len`, scaled back exactly.
- No failure channel (0.0).

### Segment-GP / ESBO-GP (`predictors/stenger/custom.py`)
- A port of `trainHierachicalESGP`/`predHierachicalESGP`: section GPs over the unique observed lengths, `mu = J_partial + Σ max(mu_q, 0)`, `var = Σσ_q²`.
- Crashed context trials are excluded from section training. The query cutoff is inserted as the leading boundary.
- The GP is `util/matlab_gp.py`: constant mean, ARD-SE, log-hyperparameters, smooth-box length-scale prior, Adam.
- `GP_COLD_REFIT_GROWTH` defaults to 0.0 (exact). **optplay and hwplay run 0.05** (`D/optplay/problems.py` `gp_cold_refit_growth=0.05`, cartpole too since 2026-08-30; `hwplay/config.py:112`). predplay/hwpred use 0. The thesis appendix ("Our port of ESBO-GP") does not mention the warm-refit approximation.
- `lsgp`: unused, drop.

---

## 6. TRAINING

### 6.1 Code as implemented (DecPFN `predictors/decpfn/train.py`, LC-PFN `predictors/lcpfn/custom.py`)

| item | DecPFN | LC-PFN |
|---|---|---|
| optimizer | `AdamW(lr, weight_decay=wd)` default betas/eps (`train.py:320`) | same (`custom.py:529`) |
| schedule | `LinearLR(start 0.01→1, 25% of steps)` then `Cosine(eta_min=1e-6)`, **stepped per optimizer step** (`train.py:326-336, 409`) | HF `get_cosine_schedule_with_warmup(epochs//4, epochs)` → warmup from 0, cosine to **0**, **stepped per epoch** (`custom.py:530, 676`) |
| epoch | `epoch_steps = 10240 // batch` (80 at batch 128) | same |
| precision | `torch.autocast(bfloat16)`, no GradScaler (`train.py:387`) | `torch.autocast(cuda)` default dtype = **float16** + `GradScaler` (`custom.py:532, 625`) |
| clip | `clip_grad_norm_(1.0)`, step skipped on non-finite norm (`:404-408`) | `clip_grad_norm_(1.0)`, step skipped on exception (`:644-656`) |
| batch | `ordered_choice(128, 256)`, fixed 128. Exact grad accumulation via OOM probe (80 GB cap) | same |
| loss | masked mean over all valid prefix positions (positions after a crash masked, `train.py:382-391`) | **unmasked** mean over `(T−k)×B` targets with a single `k ~ U{1..L-1}` per step (`:579, 626-628`). Crashed-before-k rows are kept unless env `LCPFN_MASK_CRASHED=1` (opt-in, added 2026-09-29; 2/3 = requeue variants) |
| dropout / wd / lr | searched (`config.py:43-45`) | same |
| bins | `n_bins` fixed 1000 (YAML) | same |
| stats / edges | 10k-row batch from the training chunk | 10k calibration curves from the training chunk |
| validation | 3000 held-out trajectories, every prefix position, NLL ranges `(0,99),(0,20)…(20,99),(40,99),(60,99),(80,99)` (`val.py:13-23`) | 1024 held-out curves × 32 cutoffs, scored on the final grid point only, raw-unit Jacobian, +1 window shift (`custom.py:108-110, 713-`) |
| time limit | env `TRIAL_TIME_LIMIT` seconds, checked per epoch | same |
| seeds | **none** (model init unseeded; DataLoader workers time-seeded; reservation `hash(tag)`) | same |
| outputs | `MODELS/decpfn/<ds>/<name>/{config.yaml, optim.json, trials.csv, report.pdf, <trial>/model.pt + GIFs}` | `MODELS/lcpfn/<ds>/<name>/...` + `summary.pdf` |
| loading | `resolve_model_path` (`manager/register.py:134-`): `<ds>/<name>/<trial>`, `.../best[/min\|max]/<expr>, <constraint>`, or an absolute path | same |

**HPO machinery.**
- `util/optim` is a typed search-space library (Ax 1.3 / Optuna backends; `strategy.type: grid|sobol|…`). Every trial is a numbered dir with `trials.csv`/`optim.json`.
- Stages per `writing/thesis/data/PROCEDURE.md` (`models/{decpfn,lcpfn}/cartpole/universal/`):
  - `probe`/`probe_hi` (36-point coarse)
  - `greedy/{10,100}/{1_lr,2_wd,3_dropout}`: hand-chained 1-D grids at fixed L3/E512, batch 128, head_size 16, hidden_mult 2, res 15, `output_type=linear`, `input_norm=linear`, DecPFN `extra_inputs=true`, `add_diffed_up=true`, gelu
  - `grid/{10,100}`: 20 architectures × heads, `TRIAL_TIME_LIMIT` 25 / 250 s. LC-PFN gaussian heads in separate `grid/{10,100}gauss` run 2026-09-20; DecPFN `grid/100ext`.
  - `transfer/{10,100}/<head>[_nostates|_zscore]`, the strict-minimum cell per head copied out, re-trained on each problem.
- **Pareto:** `optim.json` records a 2-objective front (NLL × training_time) for the plots. The *selection* is the scalar minimum of `val_accuracies[(20,99)]` per head (`registry.py:243-251`), not a Pareto rule.
- Resolution per problem is in the transfer `config.yaml` (HPC only). MANIFEST confirms furuta `resolution: 9`.

### 6.2 DISCREPANCY TABLE — thesis vs code

Severity: **H** = changes what is trained or evaluated; M = a stated detail differs; L = cosmetic or approximate.

| # | Topic | Thesis says (file:line) | Code does (file:line) | Sev. |
|---|---|---|---|---|
| D1 | LC-PFN crashed rows | "a trajectory that crashes before t_κ… we remove it from the current batch… optionally requeue" (method.tex:~690-700); setup.tex:153-155 "do not requeue" | Default `LCPFN_MASK_CRASHED=0`: crashed rows **stay in the loss** (their future targets are the failure atom) (`custom.py:50, 595-628`). Masking was added 2026-09-29, *after* the HPO and the linear-input transfer models. The zscore re-trains (the thesis LC-PFN) may or may not have set it. Check `model.mask_crashed` in each `*_zscore/0/model.pt` (`custom.py:~990`) | **H** |
| D2 | LC-PFN input normalization | per-channel z-score from prior samples (appendix.tex:~332-338) | Thesis checkpoints = `input_norm: zscore` of the *normalized* [0.02,1] cost ✓, **but** the HPO (greedy + grid) ran with `input_norm: linear` (min-max CDF). The zscore leaves copy those hyperparameters unchanged (`registry.py:680-695`; `lcpfn_decpfn_heads_benchmark.md:14`). The thesis presents the HPO as selecting the final configuration | M |
| D3 | LC-PFN precision | bf16 autocast for all models (appendix.tex:~355) | fp16 autocast + GradScaler (`custom.py:532, 625`) | M |
| D4 | LR schedule | cosine with linear warmup, decays to 0; warmup 25% (appendix.tex:~350-354) | DecPFN: warmup from 0.01·lr, cosine to **1e-6** (`train.py:331-333`). LC-PFN: per-**epoch** stepping, so warmup = `epochs//4` = **2 of 10 epochs (20%)** at 100k, and the LR is constant within an epoch | M |
| D5 | DecPFN grid / token count | tokens at t_ℓ = ℓ·N_stride, ℓ=1..N_grid−1 (N_grid−1 tokens), target T (method.tex eq. decpfn:grid; app tab:grid) | tokens at 0-based indices `0, s, 2s, …` < T−1 → **N_grid tokens including the very first step** (cartpole: 67 tokens at t = 1, 16, …, 991), target T (`train.py:95, 301`). The first token aggregates 1 step, not s | M |
| D6 | LC-PFN grid | t_ℓ = ⌊ℓT/N_grid⌋, ℓ=1..N_grid, excludes 0 (method.tex eq. grid) | `linspace(0, T−1, ceil((T−1)/s)+1)` → **N_grid+1 points including index 0** (cartpole 68, `custom.py:83-87`; PROCEDURE.md table) | M |
| D7 | Trajectory-TabPFN grid | same eq. grid, L points, last = T | `linspace(0, T−1, L)`, includes index 0 (`tpfn/__init__.py:79-85`) | L |
| D8 | Training size | 1×10⁵ / 1×10⁶ examples (setup.tex:140-142) | 10 / 100 epochs × 10,240 = 102,400 / 1,024,000 (`decpfn/config.py:16`) | L |
| D9 | Time encoding | t z-scored over {1..T} (LC-PFN) / {1..T−1} (DecPFN) (appendix.tex:~340-343) | z-score of the **grid index** (uniform-standardized), not of raw t. Equivalent up to the grid spacing (`architecture.py:238`; `custom.py:276-281`) | L |
| D10 | DecPFN covariates | `c_ℓ = [ΔJ, ξ_tℓ, Σξ̇]`, ξ = levels, ξ̇ = rates (method.tex eq. covariates) | features `(y, Y, j, J)`: **both** the cumulative J and the interval-summed j, `Y` levels and `y` interval-summed increments. Furuta and threetank treat every channel (incl. velocities) as a level with diff-increments. Extra channels: boiler `u`, cartpole action, furuta constant target (see P1) | M |
| D11 | DecPFN-noStates | embeds J, ΔJ, t (method.tex) | `(j, J)` + t ✓ (`train.py:375-376`) | — |
| D12 | HPO search sets | lr {1,1.8,3.2,5.6,10}e-4; wd {1e-6,1e-4,1e-2,1e-1}; dropout {0,.1,.2,.35,.5}; grid d∈{256,384,512,1024}×L∈{1,2,3,6,12}; caps 25/250 s (app tab:hpo-*) | ✓ (`PROCEDURE.md`; `gridsearch_panels.md`). Code-level search *bounds* are wider (`config.py`), but the YAML grids match | — |
| D13 | Selected configs | app tab:hpo-configs | ✓ for every cell checkable against RESULTS.md and `gridsearch_panels.md` (DecPFN 100k wd 1e-6 vs 1M wd 1e-4; LC-PFN 1M lr 3.2e-4). Per-problem leaves are not verifiable locally | — |
| D14 | Starting config | batch 128, d 512, d_ff 1024, d_k 16, L 3, linear head | ✓ PROCEDURE.md. Not stated in the thesis: DecPFN `add_diffed_up=true`, `activation=gelu`, `n_bins=1000`, `resolution=15` | L |
| D15 | Gaussian head target | z-scored with "the same per-channel mean and std used for input normalization" (appendix.tex:~372-376) | DecPFN: z-scored with the **J_T** (horizon) mean/std, not the input J statistics (`train.py:160-168`). LC-PFN: mean/std of all valid normalized curve values (`custom.py:960-966`) | L |
| D16 | Bin edges | LC-PFN on all trajectory values ✓; others on the horizon ✓; 1000 bins ✓; out-of-support targets clamped ✓ | ✓ (`custom.py:166-178`; `train.py:160-164`; `architecture.py:192-194`). Extra: LC-PFN winsorizes inputs/targets at `cap` = max calibration value | — |
| D17 | "HPO selects best cell per head" vs training-time cap | best cell within the 25/250 s budget | Grid cells that exceeded the cap *failed* (no metrics). The selected LC-PFN 1M binned L12/E384 took 250.3 s, finishing in its last epoch. Transfer re-trains on other problems ran **without** the cap | L |
| D18 | Validation sets | one shared HPO objective "validation NLL at grid cutoffs 20–99%" | DecPFN: 3000 trajectories, all prefixes. LC-PFN: 1024 curves × 32 sampled cutoffs, a different held-out sample. Both are within-family comparable only; the gridsearch figure overlays them | L |
| D19 | Seeds / reproducibility | not stated | No seeding anywhere in training. Pools regenerated time-seeded (cartpole 2026-09-20; boiler/pt2/threetank for the zscore arms) | M |
| D20 | Furuta DecPFN used on the rig | "the original" DecPFN (hw_results.tex:83-85), same 1M binned | Rig cells use `rigreset_deadtime-strat_kick0.6_binned.pt` (stratified t_d). The sim/predplay Furuta DecPFN and the ablation "dead-time+states" arm use the `rigreset(1,False)` uniform checkpoint (`hwplay/field.py:172-184`; `D/experiments/hwpred/hwpred_ladder.py:205-210`) | **H** |
| D21 | DecPFN thresholds on the rig | DecPFN / DecPFN-H fixed at τ=0.9 (sim_results.tex:400) | rig cells `mtr0.99_k06binned`/`htmr0.99_k06binned` construct `matched_threshold=0.99, fail_threshold=0.9` (`hwplay/field.py:228-306, 402-424`). `results_facts.md` calls "0.99" a name typo, but the code passes 0.99 as the matched τ | **H** (verify against `run.json`) |
| D22 | Rig budget | `N_max = 3·15·min(d,4) = 180`, `T_budget = 15·600·4 = 36000` | `hwplay/config.py` uses `D_BUDGET=5` (225/45000). The thesis trees were run with CLI overrides 180/36000. One cell (`htmr…/seed0`) ran 225/45000 and is clipped at aggregation (`aggregate.py:107,246-250`) | M |
| D23 | ESBO-GP GP refit | exact port (appendix.tex "Our Port of ESBO-GP") | `GP_COLD_REFIT_GROWTH=0.05` (warm refits between cold fits) in optplay and hwplay; exact (0.0) only in predplay/hwpred | M |
| D24 | Rig reset reopen cadence | reopen "every tenth trial" (appendix.tex:~1017) | thesis runs: `reopen_every=20` (env `HWPLAY_REOPEN_EVERY`), cycle `settled_tick_center` | L |

**LC-PFN trained only on the last input / target:** not implemented anywhere (section 5).

---

## 7. OPTIMIZERS / STRATEGIES

### 7.1 Core ESBO loop (`optimizers/bo_gp.py`)

- **Helpers**
  - `esbo_n_init = max(d, 2)` (`:80`).
  - `esbo_incumbent` (`:89`) returns +inf during the initial design, so every stop rule is disarmed there.
  - `sample_positive` (`:123`) samples N(μ, σ) truncated above the observed J_t.
- **Acquisition** is hand-rolled numpy, not Ax/BoTorch.
  - Default: min-value **MES** (`MaxValueEntropySearch(n_samples=20, grid=512)`, `:160`).
  - `ExpectedImprovement` exists but is unused.
- **`BOGPOptimizer`** (`:235/263`): `n_candidates=2000` uniform random candidates (ESBO's `resSurfRS`, no gradient optimization), `gp_iters=100`, `gp_lr=0.1`.
- **`next_theta`** (`:360`) does two GP fits:
  1. Fit on completed trials plus incomplete trials with p_fail = 0.
  2. Crash value `m1 = min(max(μ, Yopt) + 3σ, worstValid)`. Each incomplete trial gets a moment-matched mixture of crash and value, used as known per-point noise in the second fit (`:387-421`).
- `OPTIMIZER_CONDITIONING.md` documents this. Seeds: the benchmark reseeds `opt.rng` (`benchmark/optimization_execution.py:225`).
- **The `util/` path trap:** never put `util/` on `PYTHONPATH`, because `util/math.py` shadows stdlib `math`. `D/*/env.py` *appends* it.

### 7.2 Classes and thesis mapping

| thesis | sim cell (`D/optplay/optplay/field.py` CELLS `:591`) | class + args | rig cell (`P/hwplay/field.py`) |
|---|---|---|---|
| ESRS | `RandomESBO` (`:592`) | `_RandomUniformCost` (`D/.../field.py:110-162`), identical to `P/optimizers/random_stenger.py:43` `RandomUniformStengerStopOptimizer`. Merge them | `random_stenger` |
| BO | `Plain GP` (`:599`) | `PlainGPOptimizer()` (`plain_gp.py:22`) | `plain_gp` |
| ESBO-GP | `Stenger ESBO` (`:601`) | `StengerBOOptimizer(monte_carlo=True)` (`stenger.py:56`): exceed rule; fresh `StengerPredictor.predict_finals` each iteration; one positive sample used as a point mass | `stenger` |
| ESBO-Oracle | `Oracle BO` (`:809`) | `OracleBOOptimizer()` (`oracle_bo.py:117`): true J_T recovered through `ndarray.base` (`_full_arrays :52`); check `oracle_hits/misses` | — |
| DecPFN | `MTr 0.9` (`:623`, builder `:242`) | `MatchedTruncatedLivePredictionOptimizer(pred, predict_resolution=max(1, T//30), matched_threshold=0.9, fail_threshold=0.9, fail_threshold_min_t=0, monte_carlo=False)` (`live_prediction.py:912`) | `mtr0.99_k06binned`: the same class, **native cadence**, **matched 0.99** |
| DecPFN-H | `HMTr 0.9` (`:630`, builder `:213`) | `HybridMatchedTruncatedLivePredictionOptimizer(pred, matched_threshold=0.9, monte_carlo="gp", fail_threshold=0.9, fail_threshold_min_t=0)`, native cadence (`:947/1053`); per trial the lower-std of {DecPFN, fresh ESBO-GP} wins (`_prefer_stenger :1172`) | `htmr0.99_k06binned`: matched **0.99** |
| ESBO-T-TabPFN | `ESBO-TabPFN` | **No builder exists locally.** The only candidate is the archived `TabPFNBOOptimizer()` with defaults `method="ls", sample_size=25` (`field_archive.py:383`, "never run"). The thesis cell was presumably a cluster-only edit, `TabPFNBOOptimizer(method="raw", sample_size=50, version="3")`. It must be reconstructed from the cluster tree | — |
| XDecPFN-PI / FT-PFN-EI | `QC PI abs 0.99` / `QF EI abs xgrid 0.99` | `D/optplay` `q_loop.PureModelLoopOptimizer` (uses `live_prediction.condition_above`, `p_value_above`) | `qc_pi_0.99` |

**Rule semantics** (`live_prediction.py`)
- `ThresholdLivePredictionOptimizer` (`:426`): the exceed rule on every step, plus on cadence steps `p_fail > fail_threshold` (τ_crash) and the naive rule.
- `Matched*` (`:723`) sets the naive threshold to ∞ and records per-step forecasts of each trial.
- The truncated variant conditions the incumbent's reference forecast on value ≥ the incumbent's true J_T (`condition_above :130`). That is the thesis `matched` rule.
- Forecasts need ≥ `min(2, d)` observed trials (`:259`).

### 7.3 Budgets and seeds

- **Simulation** (`D/optplay/optplay/problems.py:187-255`)
  - The thesis trees use `OPTPLAY_BUDGET_MODE=capped`: `(3·15·d, 15·d·T, require_all=False)`, where the last trial is cut at `t_budget`. The default mode `full` does **not** match the thesis.
  - `budget_d` = boiler 1, threetank 2, pt2 3, cartpole 4, furuta 4 (= min(d,4)). `N_init = max(true d, 2)`, which is 8 on Furuta.
  - Seeds: `substream(problem, instance)` (FNV-1a). Per-run seed = `SeedSequence([base, phi_idx, rep])` (`benchmark/optimizer_benchmark.py:666`), which reseeds np/random/torch/opt.rng and gives a per-trial noise seed.
  - 20 screened instances per problem.
- **Rig:** `hwplay/config.py:82-130` `D_BUDGET=5` is stale. The thesis trees used 180/36000 (D22). Seeds 0–4.

### 7.4 `benchmark/` package

- `Sampler` (`sampler.py:25`)
- `PredictorBenchmark` (`predictor_benchmark.py`, 2.2k lines, notebook-driven, no CLI)
- `OptimizationExecution`/`ExecutionResult` (one run: simulated to the end, then replayed through `early_stop` at each cut)
- `OptimizerBenchmark` (`optimizer_benchmark.py:270`: factories × instances × reps, fork workers, checkpoint JSON, plots)
- `Plotter`/`PredictorPlotter`/`OptimizerPlotter` (`plotter.py`), `GlyphComparisonPlotter`

**Where the thesis results come from**
- Sim optimization: `D/optplay`, which subclasses `OptimizerBenchmark` (`run.py:57`) and uses its own aggregation.
- Sim prediction: `D/predplay`. From `P/benchmark` it uses only `predictor_benchmark._score/_ScoreTally/_DEFAULT_FLOOR` and `GlyphComparisonPlotter`.
- Rig optimization: `P/hwplay`.
- Rig prediction: `D/experiments/hwpred`.
- **No thesis number comes from `PredictorBenchmark`, `OptimizerPlotter` or `handoff_matched_live_prediction`.**

### 7.5 Keep / drop

**Keep**
- `bo_gp.py`, `plain_gp.py`, `stenger.py`, `oracle_bo.py`, `random_uniform.py`, `random_stenger.py`
- `live_prediction.py`, reduced to `LivePrediction`, `Threshold`, `Matched`, `MatchedTruncated`, `HybridMatchedTruncated`, `condition_above`, `p_value_above`
- `tabpfn_bo.py`, only if ESBO-T-TabPFN is rebuilt

**Drop**
- `frozen_stenger.py`, `gp_plus.py` (referenced by `hwplay/selfcheck.py`), `pfn_bo.py`, `stengertpfn_bo.py`, `tabpfn_surrogate.py`, `random_sobol.py`
- BTB*, `MatchedThreshold*`, `HybridMatchedLivePrediction*`
- Knobs: `virtual_source` ≠ auto, `truncate_virtual`, `virtual_std_scale`, oracle `optimism/jitter/shrink`, `use_virtual_data`, `ExpectedImprovement`

The threshold figure (`fig:app:optplay-decpfn-thresholds-violin`) also needs the sim cells `MTr 0.99 ft0.99` and `HMTr 0.99 ft0.99`.

### 7.6 `P/` symbols imported by `D/` (must keep, or re-export)

- **optimizers**
  - `optimizers.bo_gp.{is_completed, trial_final_cost, esbo_incumbent, esbo_n_init, _fit_gp}`
  - `plain_gp`, `stenger`, `oracle_bo`, `random_uniform`
  - `live_prediction.{MatchedTruncated…, HybridMatchedTruncated…, LivePrediction…, MatchedLivePrediction…, condition_above, p_value_above}`
- **predictors**
  - `predictors.decpfn.{infer.get_predictor, infer.DecPFNPredictor, architecture.load}`
  - `predictors.stenger.custom.StengerPredictor`
- **manager**
  - `manager.load.load_predictor` (→ decpfn, lcpfn, stenger, tpfn, stengertpfn)
  - `manager.pool.TrialSpec`, `manager.storage.datasets_location`
- **model**
  - `model.{system.Trial, predictor.*, optimizer.Optimizer, experiment.Experiment}`
- **benchmark**
  - `benchmark.{OptimizerBenchmark, optimizer_benchmark._final_cost/_trial_from_json, optimization_execution.*}`
  - `benchmark.predictor_benchmark._score/_ScoreTally/_DEFAULT_FLOOR`
  - `benchmark.glyph_comparison_plotter.*`, `benchmark.plotter.{PredictorPlotter, _draw_cell, …, black_style}`
- **experiments**
  - `experiments.resolve_dataset` (~50 sites), `experiments.priors.{NOISE_BASE, PHI_SPAN, THETA_BOX}`
  - module constants of boiler/threetank/pt2/cartpole/furuta systems, `experiments.furuta.hardware_benchmark.VARIANT_OF`
- **util**
  - `util.matlab_gp`, `util.mpl_styles`, `util.distributions`
- **hwplay**
  - `hwplay.{config, aggregate, figures, prediction_benchmark}`
- **Path coupling:** `D/{optplay,predplay}/*/env.py` put `../prediction` on `sys.path`. `D/c1r2` pins C++ binary paths.

### 7.7 Further thesis↔code discrepancies (optimizers)

| # | thesis | code | sev. |
|---|---|---|---|
| D25 | DecPFN virtual observation is "sampled from the horizon distribution" (method.tex single-trajectory ESBO) | `MTr 0.9` uses `monte_carlo=False`: the moment (μ, σ²) is passed to the GP as known noise. `HMTr` uses `"gp"` mode: only Segment-GP wins are sampled, DecPFN wins pass as a moment | **H** |
| D26 | single-trajectory predictors are "cheap enough to query at every timestep" | sim `MTr` forecasts every `T//30` steps. `HMTr` and the rig MTr use the native DecPFN grid | M |
| D27 | the matched rule is armed once an incumbent exists | additionally requires ≥ `min(2,d)` observed trials (`live_prediction.py:259`) | L |
| D28 | τ_crash is part of the rule | `fail_threshold=0.9` is never defined in the prose. It appears only in names (`DecPFN-M-τ-τcrash`) | M |
| D29 | ESBO-T-TabPFN at N_grid=50 | no local cell definition (see 7.2) | **H** (reconstruct) |
| D30 | ESBO-GP: GP fit as in MATLAB | additionally: unit-cube X, standardized y, clamps ℓ∈[0.01,0.5]/√(2ln10), sf∈[e⁻⁴,e], sn∈[e⁻¹⁰,e³], grad clip 100. MAP with the smooth-box prior only for noise-free fits, MLE for fits with known noise. Plus D23 (warm refits at 0.05) | L |

---

## 8. HARDWARE

- **Bridge:** `experiments/furuta/hardware.py:146` `RemoteFurutaEnv`, websocket `ws://localhost:8765`.
  - The server is `furuta/furuta/quanser-driver/remote_env_server.py` (Python 3.8 venv with the Quanser HIL SDK).
  - **The local `impl/furuta` checkout does not contain `quanser-driver`**, so `pyproject.toml`'s `gym_brt = {path="../furuta/furuta/quanser-driver"}` does not resolve. The rebuild should treat the server as external, or vendor it from the rig workstation.
  - Watchdogs: step 15 s, reset 90 s.
- **Rig system classes:** `FurutaHardwareSystem` (`hardware.py:310`) → `fast_hardware.py` cycles, ending in `SettledTickCenterFurutaHardwareSystem` (`:1342`), the thesis cycle.
  - `FurutaHardwareBenchmark` (`hardware_benchmark.py:447`), cycle selection `:659-690`, `TrialRecord` `:271`.
  - 120 Hz, T=600, θ box = `THETA_RANGE[1]` = `THETA_BOX["furuta/lqr"]`.
  - Crash \|arm\|>90°. `FurutaHoldCost(thr 5°)` + crash tail penalty (`config.py:265`).
- **Reset vs. alg:app:hw-reset:** every parameter matches `fast_hardware.py` (brake 0.4 s, coast 1.5 s, PD centring 3.0/0.2/0.35 V/2°/3 s, settle 2°/0.05 rad/s/5°/2 s dwell, virtual zero, kick 0.6 V/0.03 s, random sign). The only mismatch is the reopen cadence (D24).
- **Step rate:** one rig step is ~26 ms of websocket round trip, and `control_seconds` ≈ 7.5–9 s for 600 steps. Whether the plant is really sampled at 120 Hz (the thesis says 5 s at 120 Hz) needs checking on the server side.
- **hwplay**
  - `config.py`: seeds 0–4, `GP_COLD_REFIT_GROWTH=0.05`, `OVERLAP_REFIT` off (thesis manifests `overlap_refit: null`), env knobs `HWPLAY_RUNS`, `HWPLAY_CYCLE`, `HWPLAY_REOPEN_EVERY`, `HWPLAY_CRASH_TAIL_PENALTY`.
  - `run.py`: `python -m hwplay.run {run,status,list,pause,unpause} --methods … --seeds … --trials-budget --t-budget`. Resume replays stored trials.
  - `field.py` cells used:

    | thesis | cell | builder |
    |---|---|---|
    | ESBO-GP | `stenger` | `StengerBOOptimizer` (`:128`) |
    | BO | `plain_gp` | `PlainGPOptimizer` (`:135`) |
    | ESRS | `random_stenger` | `RandomUniformStengerStopOptimizer` (`:149`) |
    | DecPFN-H | `htmr0.99_k06binned` | `HybridMatchedTruncatedLivePredictionOptimizer(0.99, mc="gp", fail .9)` |
    | DecPFN | `mtr0.99_k06binned` | `MatchedTruncatedLivePredictionOptimizer` at native resolution |
    | XDecPFN-PI | `qc_pi_0.99` | imports `D/optplay` (`model_specs`, `q_loop.PureModelLoopOptimizer(0.99, "PI")`, `q_predictor.QPredictor`) via `sys.path` (`:318-376`); needs env `C1R2_PROBLEM=furuta` |

  - Output: `<HWPLAY_RUNS>/lqr_hardware/{manifest.json, <cell>/seed<k>/{run.json, trials.jsonl}}`. The `trials.jsonl` lines hold θ, outcome, J, Y (t×5), every timing and the reset diagnostics.
- **Thesis run dirs:** `hwplay/runs_holdcost5_newbox_reset_fixed` (**without weight**, 121 MB) and `…_different_plant` (**with weight**, 114 MB). There are 8390 trial lines.
  - The two have identical manifests. The weight is identified **only** by the directory (`D/thesis_figures/hwplay_optimization.py:139-145`).
  - Drop `qc_pi_0.99_n10k` (not in the thesis).
  - Obsolete trees: `runs`, `runs_dtunif20`, `runs_holdcost5`, `runs_holdcost5_newbox`, `runs_holdcost5_newbox_different_pendulum`, `runs_step30` (~700 MB).
- **Rig reset models:** `furuta_rigreset_models/MANIFEST.{md,json}`, 12 DecPFN checkpoints = {nodeadtime, deadtime-strat, deadtime-unif} × kick{0.6,0.2} × {binned, nostates}, all 100 epochs, resolution 9. Exported by `scripts/export_rigreset_models.py`.
  - Needed: `deadtime-strat_kick0.6_binned` (rig optimizer + hwpred DecPFN) plus the four ablation arms `deadtime-unif_kick0.6_{binned,nostates}` and `nodeadtime_kick0.6_{binned,nostates}`.
  - **No `.pt` is present locally.** Get them from the HPC `models/decpfn/furuta/...` or the rig workstation `/home/tony/.../furuta_rigreset_models/`. XDecPFN: `/work/dln18500/c1r2/furuta_lnJ_W128_qw05_successgate/ckpt_final/f_s20260931_seg6_final.pt`.
- **Hardware prediction benchmark (thesis):** `D/experiments/hwpred/hwpred_ladder.py` (+`.slurm`, `hwpred_timing.py`). It is **not** `P/hwplay/prediction_benchmark.py` and **not** `P/scripts/furuta_hardware_predictions*`, both of which are obsolete.
  - Ladder 90/45/22/11/5/2/1/0, cutoffs 10–90% of the trial's own t, GP growth 0.
  - Results in `D/experiments/hwpred/results/` (1.5 MB). The thesis CSV comes from `D/thesis_figures/hwpred_benchmark.py`.
- **Old data to copy:**
  - the two thesis run trees (~235 MB)
  - `D/experiments/hwpred/results/*`
  - `furuta_rigreset_models/MANIFEST.*`
  - `c1r2_furuta_lnJ/*`
  - `writing/thesis/data/sources/{hwpred_benchmark.csv, fig1_cross_5prob_canonmed.csv, cluster_answers_2026-09-24.md}`
  - the 6 checkpoints above
- **Hardcoded:** `/home/tony/...` (`hwplay/field.py:52`, run.json, `hwplay/logs/launch_*.sh`), `/work/dln18500/...`, `ws://localhost:8765`, USB vendor `21df` (`reset_qube.py:40-52`), sibling layout `REPO.parent/"dynformer"`.

---

## 9. SLURM / HPC

- **`manager/base.slurm`**
  - 1 node, 8 CPUs, 32G
  - `module load GCCcore/13.3.0 Python/3.12.3` (+ `cuda/12.4` if `USE_CUDA=1`; that is an **invalid** module name per the root `CLAUDE.md`, harmless because torch cu128 bundles CUDA)
  - `OPTIM_DISABLE_USETEX=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `PATH=$HOME/.local/bin`
  - `flock $HOME/.cache/uv_sync.lock uv sync --frozen --no-dev` → `uv run --no-sync ./manager/load.py "$@"`
- **Accounts:** CPU `-A rwth2055` (`c23ms/c23mm/c23ml`), GPU `-A thes2287` (`c23g`) (root `CLAUDE.md`). These appear in `tools/*/gen.slurm`, `validate.slurm` and `scripts/*.slurm` (`#SBATCH --account`/usage lines). `load.py` takes the account only from `-A`.
- **Modules:** `GCC/13.3.0` (C++), `imkl/2024.2.0` (furuta), `GCCcore/13.3.0 Python/3.12.3`, `CUDA/12.8.0` (some scripts).
- **Env-specific items to make configurable:** `$HPCWORK`/`$DATA`, `$MODELS` (defaults to the relative `./models`), `$HOME/.local/bin`, `/rwthfs/rz/cluster/home/dln18500/...` (`scripts/prior_v*_final.sh`), `dln18500` default user (`tools/cartpole_cpp/validate.py:59`), `/work/dln18500` (D side), `/home/tony` (rig), partitions, accounts, `OMP/MKL_NUM_THREADS=1`.
- **Operational gotchas worth keeping in a README:**
  - A bare python segfaults without the modules loaded.
  - The login node SIGTERMs runs longer than ~30 min.
  - `training_time` is not comparable across sessions (63.5 s / 94.7 s / 139.1 s for the same cell), so the 25/250 s caps are session-dependent.
  - GPU MIG ordinal trap on `login23-g-1`.

---

## 10. TESTS

**There is no test suite.** No `tests/`, no `test_*.py`, no pytest config.

Ad-hoc validators exist:
- `tools/furuta_cpp/validate.py` and `tools/cartpole_cpp/validate.py` (C++↔Python parity)
- `D/experiments/f6/*` and `D/experiments/boiler_port/*` (plants parity)
- `hwplay/selfcheck.py` (rig field sanity)
- `Experiment.test_{failed,success}_percentage` (`model/experiment.py:135-145`, rate measurement, not tests)

The rebuild should turn the parity checks and a few golden-trajectory checks into pytest.

---

## 11. DEPENDENCIES

- Python `3.12` (`.python-version`; pyproject `>=3.12,<3.14`). Manager: `uv` with `uv.lock`.
- Locked versions (`uv.lock`):

| package | version |
|---|---|
| torch | 2.11.0+cu128 (linux, index `pytorch-cu128`) / 2.12.1 (mac) |
| tabpfn | 8.1.0 |
| tabpfn-client | 0.3.3 |
| ax-platform | 1.3.1 |
| botorch | 0.18.1 |
| gpytorch | 1.15.2 |
| numpy | 2.4.6 |
| scipy | 1.18.0 |
| pandas | 2.3.3 |
| optuna | 4.9.0 |
| gymnasium | 1.3.0 |
| mujoco | 3.10.0 (unused by the kept code) |
| matplotlib | 3.11.0 |

- Also listed: `dataclasses-json`, `scikit-learn`, `websockets>=13`, `pillow`, `mediapy`/`moviepy`/`pygame`/`pyvirtualdisplay` (GIF/video only → drop), `gym_brt` (path dep, see section 8), `setuptools<70`.
- C++: g++ 13.3 (C++17), Intel MKL 2024.2 for furuta (LAPACK dgges/dtgsen), x86-64-v3 target.
- TabPFN v3 checkpoint file (download).

---

## 12. OPEN QUESTIONS for Tony

1. **D1 LC-PFN crash handling.** Should the canonical LC-PFN drop crashed-before-κ rows (as the thesis states, `LCPFN_MASK_CRASHED=1`) or keep them (the code default)? First check `model.mask_crashed` in the HPC `lcpfn/*/transfer/*/*_zscore/0/model.pt`. If 0, either fix the thesis text or re-train.
2. **Canonical LC-PFN input norm.** Use `zscore` (the thesis checkpoints) with HPO values found under `linear` (D2)? Or re-run the LC-PFN HPO under zscore?
3. **LR schedule / precision (D3, D4).** Make both models identical (per-step warmup from 0, cosine to 0, bf16), as the thesis states, or keep each model's code behaviour so that the existing checkpoints are reproduced bit-for-bit (as far as unseeded training allows)?
4. **Grids (D5–D7).** Adopt the thesis grid definitions (no index-0 token, N_grid−1 DecPFN tokens), which means re-training, or keep the code grid and correct the thesis equations/tables?
5. **DecPFN feature set (D10, P1).** Keep the extra channels (boiler `u`, cartpole action, furuta target≡0, `Y`+`y` for every channel), or match `tab:app:problems`?
6. **Virtual observations (D25) and ESBO-T-TabPFN (D29).** Canonical = code (moment pass-through for MTr) or thesis (sampled)? Who has the cluster `ESBO-TabPFN` cell definition (method, sample_size, version)?
7. **Furuta on the rig (D20, D21).** Which checkpoint is "the" Furuta DecPFN: stratified (rig) or uniform (sim and ablation)? Confirm the rig cells' matched τ from `runs_*/lqr_hardware/mtr0.99_k06binned/seed0/run.json` (0.99 per `field.py`, vs τ=0.9 in the thesis).
8. **Budgets / GP knob (D22, D23).** Hardcode 180/36000 for the rig and `GP_COLD_REFIT_GROWTH=0.05` for optplay/hwplay? Should the thesis mention the warm-refit approximation?
9. **Reproducibility policy.** Introduce seeds (pool gen, reservation, model init, `random`/`np.random` in the simulators) in the new repo, accepting that the old checkpoints cannot be regenerated bit-for-bit?
10. **plants pools in C++?** Add a `gen` subcommand to `plants_cpp` to make "all priors in C++" true, or change the sentence?
11. **Generation-speed table.** No measuring script exists. Write one, and accept that the numbers may shift?
12. **Which artefacts become "old data".** The two hwplay trees, the D-side predplay/optplay/hwpred result files, and which checkpoints (all `transfer` leaves for 5 problems × {decpfn, decpfn_nostates, lcpfn_zscore} × {10,100} × 3 heads, or only the E6-binned ones)? Also whether to archive the HPC `grid/`/`greedy/` `trials.csv` (needed to regenerate `gridsearch_panels` and the HPO tables).
13. **Drop list confirmation:** `lsgp`, `training/lcpfn`, `handoff_matched_live_prediction`, `notebooks`, all non-`universal` variants, `furuta/pid`, `furuta/lqr{,-realistic,-new}/universal`. Keep `furuta/lqr-new/rigreset` only for the ablation's no-dead-time arm.
14. **`results_facts.md`** must be updated to `lcpfn_zscore/...` before it is used as a spec for the rebuild.
