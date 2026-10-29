# Strategies and the optimization loop

Code: `src/esbopfn/strategies/` (`base.py`, `loop.py`, `rules.py`, `gp_based.py`, `pure_loop.py`).
Thesis references: `B` = background.tex, `M` = method.tex, `S` = setup.tex (line numbers approximate).

## Usage

```python
from esbopfn.problems import get_problem
from esbopfn.strategies import make_strategy, run_optimization, strategy_rng

problem = get_problem("cartpole")
strategy = make_strategy("decpfn", problem, strategy_rng(seed), tau=0.9)
record = run_optimization(problem, instance, strategy, seed)   # -> RunRecord
blob = record.to_json()
```

`make_strategy(id, problem, rng, **params)` takes the thesis defaults and applies overrides.
`predictor=<object>` replaces the model resolved from `model` (used by the tests with fakes).

## The loop (`run_optimization`, alg:ct:esbo, B:770-814, S:41-58, M:84-90)

- Budgets: `N_max = 3·15·min{d,4}` trials and `T_budget = 15·T_max·min{d,4}` steps, both applied
  (`n_max`, `t_budget` arguments override them).
- Initial design: `N_init = max{d,2}` iid uniform controllers, run without early stopping.
- Main loop while `n < N_max` and `Σ T_i ≤ T_budget − T_max` (a full trial must still fit). `T_i`
  is the observed length of trial `i`, the crash step for a crashed trial.
- Each trial is simulated in full once with `substream(seed, n)`. The stop rules are then replayed
  over the logged steps, which is equivalent to stepping the plant because every decision at step
  `t` reads only the prefix up to `t`. At a step `t < T_max` a crash ends the trial first, then the
  `exceed` rule (`J_t ≥ J⁻_n`, for every strategy except BO), then the strategy's own rule at its
  `decision_steps()` on the trial cut at `t`.
- Incumbent `θ⁻_n`, `J⁻_n`: the best successful complete trial (B:610-612, success per M:404-407).
  `J⁺_n`: the largest total cost among complete trials. Both are refreshed after every trial.
- Stop rules are armed only after the initial design and once an incumbent exists.

### RunRecord

`RunRecord(problem, strategy, params, seed, instance_seed, T, n_init, n_max, t_budget, trials,
incumbent_index, incumbent_J, timing, wall_s, version)`. Each `TrialRecord` holds `theta, t_obs,
J_obs (inf if crashed), crashed, success, complete, J_true (total cost of the uncut trajectory),
source ("init" | "strategy"), propose_s, stop_rule_s, end_s, J` (the observed curve
`J_1..J_{t_obs}` as float32 values when `record_curves=True`, else `None`). `incumbent_index` and
`incumbent_J` are the incumbent after each trial (`-1` and `inf` while none exists). `timing` sums the
strategy's work per category: `gp_acq` (objective GP fits and MES), `segment_gp`, `tabpfn_forecast`,
`model_forecast` (transformer predictions for stop rules and virtual observations), `model_acq`
(pure loop candidate predictions and acquisition). `to_json()` writes compact JSON with infinite
costs as `Infinity`, `from_json()` reads it back.

## Stop rules (`rules.py`)

| rule | definition | reference |
|---|---|---|
| `exceed` | stop as soon as `J_t ≥ J⁻_n` | B:638-644 |
| `naive` | stop if `Pr[J̃ ≥ J⁻_n] > τ`, the crash mass counts as `≥` | M:855-873, eq:method:single-trajectory:early-stopping-naive |
| `matched` | stop if `Pr[J̃(θ_{n+1}) ≥ J̃(θ⁻_n) \| J⁻_n ≤ J̃(θ⁻_n)] > τ`, both predicted at the same cutoff from their own prefixes | M:884-906, eq:method:single-trajectory:early-stopping-matched |

`prob_geq` is exact for two bar distributions (union of the edges, ties inside a shared interval
split in half), Gaussian predictions are discretized into 512 bars over ±8 standard deviations. In
`matched` the incumbent prediction is conditioned on the event `J ≥ J⁻_n` of the full mixture
(`truncate_below`, the crash atom belongs to that event). `crash_threshold` (default `None`) adds a
stop when the running trial's crash probability exceeds it, only for the appendix variants.

## Strategies

| id | strategy | parameters (defaults) |
|---|---|---|
| `esrs` | θ ~ U(Θ), `exceed` (S:598-603) | none |
| `bo` | objective GP + MES, no early stopping (S:604-605) | `cold_refit_growth=0.05`, `n_candidates=2000`, `gp_iters=100` |
| `esbo-gp` | Segment-GP virtual observations, `exceed` (B:665-814) | as `bo` |
| `esbo-oracle` | virtual observation = true total cost (S:606-610) | as `bo` |
| `esbo-t-tabpfn` | Trajectory-TabPFN replaces the Segment-GPs, `exceed` (M:325-361) | as `bo`, `L=50`, `model="tabpfn/<L>"`, `predictor` |
| `decpfn` | `exceed` + `matched`, DecPFN virtual observations (M:908-935) | as `bo`, `tau=0.9`, `crash_threshold=None`, `model="decpfn/<p>/binned/1M"`, `predictor` |
| `decpfn-h` | as `decpfn` with `hybrid` virtual observations (M:913-927) | as `decpfn` |
| `xdecpfn-pi`, `xdecpfn-ei` | pure model loop (M:1146-1182) | `acq`, `tau=0.99`, `crash_threshold=None`, `model="xdecpfn/<p>/default/default"`, `predictor`, `n_uniform=400`, `n_local=1600`, `top_k=5`, `local_scale=0.10` |
| `ftpfn-ei`, `ftpfn-pi` | pure model loop | as above, `model="ftpfn/<p>/default/default"` |

Display names come from `esbopfn.plots.style.DISPLAY_NAME` (`StrategySpec.display_name`).

### GP strategies (`gp_based.py`)

`GPStrategy.propose` follows alg:ct:esbo:
1. Virtual observations `E_n` for every early-stopped trial, drawn anew in every iteration from the
   strategy's predictive distribution conditioned on `J̃ ≥ J_{T_i}` (M:129-140). A finite draw is the
   virtual observation, an `∞` draw sends the trial to the crash rule.
2. For crashed trials (and `∞` draws) the objective GP is fit on the complete trials `O_n` and
   `J_virt = min{max{μ_O(θ), J⁻_n} + 3σ_O(θ), J⁺_n}` (eq:esbo:crashed-virtual-objective, B:727-748).
   `σ_O` includes the learned observation noise, as in the MATLAB port. The fit on `O_n` is skipped
   when there is nothing to impute (it consumes no randomness, so this changes nothing else).
3. The objective GP is fit on `O_n ∪ E_n ∪ F_n` and the next controller is the maximizer of max-value
   entropy search (`mes_acquisition`, incumbent `J⁻_n`) over 2000 uniform random candidates.

The objective GP is `predictors.gp.ExactGP`. The fit on `O_n` and the fit on `D_virt` are two
separate hyperparameter chains for `cold_refit_growth`.

- `esbo-gp`: `predictors.segment_gp.SegmentGP` on the whole history, sampled with `sample_positive`
  (Gaussian truncated at `J_{T_i}`).
- `esbo-oracle`: `on_trial_end` hands the strategy the uncut trajectory and the virtual observation is
  its true total cost (point mass). If the uncut trajectory crashes, the crash rule applies.
- `esbo-t-tabpfn`: `tabpfn/50` predicts all early-stopped trials from the history in one fit, the
  bar prediction is sampled above `J_{T_i}`.
- `decpfn`: the matched rule is evaluated at every step of `predictor.predictable_steps` (the decoder
  grid). The running trial's prediction is computed at each of those steps, also during the
  initial design, so the incumbent's predictions at every cutoff are reused from its own run (only
  the current incumbent's curve is kept). The virtual observation of an early-stopped trial is
  sampled from its static prediction at the last grid token `≤ T_i`, conditioned on `J ≥ J_{T_i}`.
- `decpfn-h`: every iteration and for every early-stopped trial, the updated Segment-GP Gaussian and
  the static DecPFN prediction conditioned on `J ≠ ∞` (its value-part variance) are compared and the
  lower-variance one is sampled. The chosen source is in `strategy.last_sources`.

### Pure model loops (`pure_loop.py`)

No GP. Proposal: the bar prediction `p(· | θ*, H_n)` of every candidate from
`predictor.predict_unrun(thetas, history)` (batched), closed-form acquisition in raw `J` with
`c_k = max(min(J⁻_n, b_k), b_{k−1})`:

- EI: `(1−w_∞) Σ_k w_k (c_k − b_{k−1})(J⁻_n − (b_{k−1}+c_k)/2)/(b_k − b_{k−1})` (eq:method:pure-loop:ei)
- PI: `(1−w_∞) Σ_k w_k (c_k − b_{k−1})/(b_k − b_{k−1})` (eq:method:pure-loop:pi)

Candidates (S:104-115): 400 uniform plus 1600 Gaussian draws, each around one of the five best
complete controllers chosen uniformly, standard deviation `0.10 ×` the box extent per dimension,
clipped to Θ. Stopping: `exceed` and `naive` with `predictor.predict(running, history)`, evaluated at
the decoder grid token steps for both XDecPFN and FT-PFN (S:97-102).

## Specification choices (thesis silent)

- Initial design and the fallback design are iid uniform draws from `substream(seed, "init")`, so
  every strategy gets the same initial design for the same seed. Per-trial simulation randomness is
  `substream(seed, n)`. Strategies use `strategy_rng(seed) = substream(seed, "strategy")`.
- If no successful complete trial exists after the initial design (the thesis assumes one does,
  B:768), the loop keeps drawing uniform controllers without early stopping until one exists. This
  holds for every strategy, the pure loops included (as in the thesis runs).
- The initial design always runs in full, even if `N_init` full trials exceed a user-set `t_budget`.
- `J⁺_n` counts complete trials, `J⁻_n` successful complete trials (identical without a success
  criterion).
- `exceed` is only evaluated at `t < T_max`, a trial whose cost reaches `J⁻_n` at the horizon is
  complete. A crash at step `t` takes precedence over a stop at the same step.
- DecPFN trials stopped (by `exceed`) before the first grid token have no DecPFN prediction and get
  a Segment-GP virtual observation, in `decpfn` and `decpfn-h` alike.
- `decpfn-h` does not use a Segment-GP prediction of zero variance (no segment GP could be fit),
  ties go to DecPFN.
- `matched` conditions the incumbent prediction on `J ≥ J⁻_n` as a whole mixture (the literal
  conditioning event of the thesis equation).
- `cold_refit_growth = 0.05` (deviation D2 in [DEVIATIONS.md](DEVIATIONS.md)).
