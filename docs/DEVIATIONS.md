# Deviations and specification choices

The thesis text is the specification of this code. This page lists every place where the code
differs from the text (first section) and every choice the code makes where the text is silent,
ambiguous or gives only the idea (second section). Each item names the thesis place it refers to
(section titles and `tab:`, `fig:`, `eq:`, `alg:` labels of the thesis) and links the doc that
describes the implementation. Any other difference from the thesis text is a bug.

## Deviations from the thesis text

| # | Item | Thesis | What the code does |
|---|---|---|---|
| D1 | Gradient clipping of XDecPFN and FT-PFN | Appendix "Training of the Predictors": all four models clip the gradient norm at 1 | XDecPFN and FT-PFN train without clipping (`clip_grad_norm: null` in their hparams files). LC-PFN and DecPFN clip at 1. |
| D2 | GP refit approximation `cold_refit_growth` | Appendix "Our Port of ESBO-GP" fits the GP hyperparameters with Adam and says nothing about warm starts | An `ExactGP` refits from scratch only once its training set has grown by 5 % since the last cold fit and otherwise runs 15 % of the steps from the previous hyperparameters. Optimization runs use `cold_refit_growth = 0.05`, and `0.0` makes every fit cold ([PREDICTORS.md](PREDICTORS.md#exact-gp-esbopfnpredictorsgp)). |
| D3 | Training set sizes of the 100k and 1M models | tab:setup:model-cost, "10⁵ / 10⁶ trajectories" | Exactly 100,000 and 1,000,000 distinct trajectories, one pass each, `floor(N/128)` optimizer steps at batch 128 (781 and 7,812) ([TRAINING.md](TRAINING.md#the-thesis-recipe)). |

## Specification choices

### Problems and priors ([DATA.md](DATA.md), `src/esbopfn/problems/README.md`)

| Item | Thesis | Choice |
|---|---|---|
| PT2 input dead time | tab:app:problems, "0.1 s input dead-time" | Exactly 10 simulation steps of 0.01 s. |
| Sensor noise | subsubsec:setup:prior-instantiation, tab:app:prior-noise | `σ_i = base_i · S · η_i` with one `S ~ LU(1.5/4, 4/1.5)` per instance and one `η_i ~ LU(1/1.5, 1.5)` per noise group, on every problem including the single-channel ones. Furuta draws one `η` for both angles and one for both rates. |
| Parameter estimate `p̂` | subsubsec:setup:prior-instantiation, `p̂ = max{0, N(p, (p/10)²)}` | Applied literally, a negative draw is clipped to 0 and not redrawn. Only Furuta uses `p̂` (for the LQR design). |
| State channels | tab:app:problems, the `x_t` rows | Exactly those channels: Boiler `[temperature]`, Three-Tank `[V2, V3, V4]`, PT2 `[y]`, Cartpole `[x_cart, phi_pole]` with rates `[xdot_cart, phidot_pole]`, Furuta `[phi_arm, phi_pend]` with rates `[phidot_arm, phidot_pend]`. |
| Furuta dead time and kick | tab:app:prior-phi | Dead time `U[8.33, 25] ms` and kick amplitude `U[0, 0.6] V` per instance, kick sign drawn per trial, 30 ms pulse. |
| Grids | subsec:setup:hyperparameters, tab:app:grid | Strides are derived, not tabulated: Cartpole stride 15 gives `N_ref = 67`, every other problem takes the stride whose `ceil(T/stride)` is closest to 67 (ties to the smaller stride). This yields the table's 11, 7, 8, 15 and 9. |
| Rejection rate `ρ` of the single-trajectory prior | subsubsec:method:prior-single | The unsuccessful rate comes from a pilot of 50,000 unscreened draws with the fixed seed 20261028, so every dataset of a problem uses the same rate. |
| Rates of the trajectory-set prior | subsubsec:method:prior-sets | The crash rate and the unsuccessful-given-not-crashed rate of an instance come from the first runs of its 100 screen controllers. After `1000 + 1000 n` draws the remaining slots accept any trajectory, recorded as `forced`. |
| Degeneracy screen design | subsec:method:prior, "100 random controllers" | The C++ screen draws the 100 controllers from a Sobol' design with a random digital shift per instance, the Python screen draws them iid uniform. A controller stops at its first unsuccessful repeat, and the screen stops once 25 passes are out of reach. |
| Generator random streams | appendix "Sampling Cost", priors implemented in C++ | C++ and Python share the distributions, not the random streams. Chunk `k` of a dataset is seeded with `derive_seed(seed, k)`, so the worker count never changes a dataset. |

### Predictors and training ([TRAINING.md](TRAINING.md), [CROSS_TRAJECTORY.md](CROSS_TRAJECTORY.md))

| Item | Thesis | Choice |
|---|---|---|
| Calibration sample sizes | appendix "Training of the Predictors", statistics "estimated from samples of the prior" | LC-PFN and DecPFN: 20,000 pool rows for the z-score statistics and up to 100,000 horizon values for the DecPFN bar edges. XDecPFN: the first 2,000 training instances. FT-PFN: the first 20,000 training contexts. |
| Validation sets | subsec:setup:hyperparameters | DecPFN 3,000 rows, LC-PFN the first 1,024 of them, XDecPFN 256 held-out instances with 16 examples each, FT-PFN 2,000 held-out contexts, all from their own seed streams. |
| Validation cutoffs | subsec:setup:hyperparameters, "grid cutoffs between 20 % and 99 %" | `κ = round(0.2 L) + 1 .. round(0.99 L)` over the `L = N_grid − 1` possible cutoffs, NLL in raw cost units over every (row, cutoff) pair not yet crashed. |
| HPO wall-clock cap | tab:app:hpo-stages | Stage 1 runs uncapped. In stage 2 the 25 s or 250 s cap counts training time only, and a capped run follows its full learning-rate schedule until it is cut. |
| XDecPFN cost inputs | subsubsec:method:xdecpfn, subsec:setup:hyperparameters "normalizes and bins ln J" | The cumulative cost enters as `ln J` and the bins are over `ln J_T`, while the increment `ΔJ` stays in raw cost units. The controller is z-scored with the moments of the uniform prior on the search box. |
| FT-PFN cost inputs | subsubsec:method:ftpfn | Raw `J` values are z-scored and clamped into the range of the calibration samples. |
| FT-PFN batching | subsubsec:method:ftpfn | Contexts are packed into rows by first-fit decreasing, and the PFN block mask is a boolean mask of `scaled_dot_product_attention`. |

### Strategies ([STRATEGIES.md](STRATEGIES.md#specification-choices-thesis-silent))

| Item | Thesis | Choice |
|---|---|---|
| Initial design | alg:ct:esbo, "sampled uniformly from Θ" | `N_init = max(d, 2)` iid uniform controllers from `substream(seed, "init")`, shared by every strategy for the same seed and always run in full. |
| No successful initial trial | background, the assumption that one initial trial succeeds | Uniform controllers without early stopping are drawn until a successful complete trial exists, for every strategy. |
| GP acquisition candidates | appendix "Our Port of ESBO-GP", MES | 2,000 uniform random candidates for BO, ESBO-GP, ESBO-Oracle, ESBO-T-TabPFN, DecPFN and DecPFN-H. |
| `exceed` at the horizon | the `exceed` rule of the background chapter | Checked at `t < T` only, a trial that reaches `J⁻_n` at the horizon is complete. A crash takes precedence over a stop at the same step. |
| `matched` conditioning | eq:method:single-trajectory:early-stopping-matched | The incumbent prediction is conditioned on `J ≥ J⁻_n` as a whole mixture, the crash atom belonging to that event. |
| DecPFN before its first grid token | subsubsec:method:single-trajectory-esbo | A trial stopped by `exceed` before DecPFN's first grid token gets a Segment-GP virtual observation, in `decpfn` and `decpfn-h`. |
| DecPFN-H ties | subsubsec:method:single-trajectory-esbo | A Segment-GP prediction of zero variance is not used, and ties go to DecPFN. |
| Pure loop stop grid and acquisition | subsec:setup:implementation, subsubsec:method:cross-trajectory-esbo | XDecPFN and FT-PFN loops evaluate stops at the decoder grid token steps. EI and PI are computed in raw `J` on the predictor's edges (XDecPFN maps its `ln J` edges back). |
| `crash_threshold` | appendix "Per-Problem Optimization Results" (`τ_crash` variants) | Off by default, set only for the appendix threshold variants. |

### Benchmarks ([BENCHMARKS.md](BENCHMARKS.md#specification-choices))

| Item | Thesis | Choice |
|---|---|---|
| Cutoff snapping | subsubsec:setup:prediction-metrics, "snapped to the closest grid point before" | The last predictable step at or before `f · T`. When none exists (FT-PFN at 10 %), the first predictable step is used and the row is flagged `cutoff_fallback`. On the rig the cutoffs refer to the query's own length and XDecPFN drops cutoffs before its first grid point. |
| Seeds | not stated | `BENCH_SEED = 20261029`, from which instances, fields and optimization runs derive their seeds through `substream`. |
| Field budgets | subsubsec:setup:prediction-tasks | Oracle fields override the budgets so that each run has exactly its number of trials. |

### Hardware ([HARDWARE.md](HARDWARE.md#the-reset-between-trials))

| Item | Thesis | Choice |
|---|---|---|
| Connection reset cadence | app:hw-reset, "after every tenth trial" | At every trial index that is a positive multiple of 10, placed after the brake step: the arm is centered and settled, its position is read, the connection and the board are reopened, and the read position is carried across the reopen. |
| Arm zeroing order | alg:app:hw-reset | The arm zero is set after the settle loop and before the kick. Because a board reopen re-zeroes the encoder where the arm stands, the session keeps its own arm frame and the trial zero is taken in that frame. |
| Reset thresholds | alg:app:hw-reset, "small for 0.4 s", "stuck", "keeps moving away" | Brake until `\|φ̇_pend\| < 0.2 rad/s` within 3° of hanging over 0.4 s, stuck means `\|φ̇_arm\| < 0.05 rad/s`, moving away means more than 8° further from the center, and after 25 s the trial starts anyway with `settled: false`. |
