# Thesis DecPFN and LC-PFN checkpoints

The DecPFN, DecPFN-no-St and LC-PFN weights of the earlier implementation can be imported into
the model tree and served through the usual registry. They keep the old architecture, features,
grids and scaling: small legacy classes reproduce the old forward pass with the original
parameter names and load the weights strictly (`src/esbopfn/predictors/decpfn/legacy.py`,
`src/esbopfn/predictors/lcpfn/legacy.py`). The XDecPFN and FT-PFN checkpoints follow the same
pattern ([CROSS_TRAJECTORY.md](CROSS_TRAJECTORY.md#thesis-checkpoints)).

## 1. Export in the old environment

The LC-PFN checkpoints are pickled whole modules that only unpickle with the old class paths, so
a script that runs in the old environment writes them into a neutral format first:

```bash
OLD=~/bachelorarbeit/impl/prediction   # checkout of the earlier implementation
PYTHONPATH=$OLD:$OLD/util $OLD/.venv/bin/python tools/export_thesis_single_trajectory.py export \
    --models ~/Downloads/dynformer_timing_bundle/repo/prediction/models \
    --out ~/Downloads/thesis_single_trajectory_export
```

It walks `<models>/<decpfn|lcpfn>/<dataset>/transfer/<10|100>/<head>[_nostates][_zscore]/0/model.pt`,
loads every checkpoint with the old code and writes `<out>/<family>/<problem>/<variant>/<size>/`:

| file | content |
|---|---|
| `state_dict.pt` | plain tensors (DecPFN: network plus `head.*`, LC-PFN: the module without its `criterion.*` buffers) |
| `arch.json` | every hyperparameter needed to rebuild the old module (width, layers, heads, FFN width, head type, bins, feature names and order, stride, sequence length, input normalization, LayerNorm placement, activation) and the old training config |
| `calib.json` | DecPFN: input mean and std, input quantile edges, head edges in raw units, Gaussian target mean, std and range. LC-PFN: the old normalization spec (`cap`, `[0.02, 1]`), input normalization edges, bar borders in model space and raw units, crash threshold, Gaussian target statistics, `mask_crashed` |
| `source.json` | old path, `SOURCE.txt` content (if present), SHA-256, file date |

`<out>/manifest.json` lists every exported, skipped and failed leaf. Name mapping:

| old | new |
|---|---|
| `decpfn/<ds>/.../<head>` | `decpfn/<p>/thesis-<head>/<size>` |
| `decpfn/<ds>/.../<head>_nostates` | `decpfn-nostates/<p>/thesis-<head>/<size>` |
| `lcpfn/<ds>/.../<head>_zscore` (the thesis LC-PFN) | `lcpfn/<p>/thesis-<head>/<size>` |
| `lcpfn/<ds>/.../<head>` (linear input normalization) | `lcpfn/<p>/thesis-<head>-linearnorm/<size>`, only when the `_zscore` leaf of the same problem, size and head is absent |
| `transfer/10`, `transfer/100` | size `100k`, `1M` |
| `<p>/universal`, `furuta/lqr-new-realistic/rigreset(1,False)` | `<p>`, `furuta` (`furuta/lqr` is skipped) |

The timing bundle holds 88 such leaves: DecPFN and DecPFN-no-St with the binned, Gaussian and
linear heads at both sizes on all five problems (60), and LC-PFN with linear input normalization
(28, Furuta without the Gaussian head). It has no `_zscore` leaf, so every LC-PFN is imported as
`thesis-<head>-linearnorm`. The `_zscore` leaves import under `thesis-<head>` as soon as an
export of them is available.

`tools/export_thesis_single_trajectory.py parity --models DIR --out tests/fixtures/legacy_single_trajectory`
writes the parity fixtures of the tests (below).

## 2. Import

```bash
pfn import thesis-models --from ~/Downloads/thesis_single_trajectory_export [--problem cartpole] [--overwrite]
```

`import_thesis(from_dir, *extra_dirs, home=None, problems=None, overwrite=False)` in
`esbopfn.benchmarks.import_models` finds the export leaves (`*/*/thesis-*/*/arch.json`) next to the
XDecPFN and FT-PFN layouts, so one call can take the export and the bundle directories together.
`import_single_trajectory_leaf(leaf)` imports one leaf. Each model directory gets `model.pt`
(state dict plus `arch.json` as the model config), `calib.json` and a `config.yaml` with
`provenance: thesis`, `legacy: true`, the old path, its SHA-256, the old dataset and config, the
epochs (10 or 100) and the old optimizer steps (800 or 8,000). Ids whose variant starts with
`thesis` resolve to the legacy predictors:

```python
from esbopfn.predictors import get_predictor
dec = get_predictor("decpfn/cartpole/thesis-binned/1M", "cartpole")
nos = get_predictor("decpfn-nostates/furuta/thesis-binned/1M", "furuta")
lc = get_predictor("lcpfn/cartpole/thesis-binned-linearnorm/1M", "cartpole")
lc.predict_curve(trial)
```

Without the import, `get_predictor` raises `ModelNotTrained` with the import hint.

## 3. The legacy predictors

Both implement the `Predictor` protocol (context-blind, no unrun queries), return
`BarPrediction` with the old raw-unit edges or `GaussianPrediction`, with `p_crash` the crash
output, and the known outcome for crashed and complete queries.

- `LegacyDecPFNPredictor` feeds the old tokens at the 0-based steps `0, s, 2s, ... < T - 1` that
  lie before `t_obs` and reads the prediction at the last one. `predictable_steps` are the 1-based
  token steps `1, 1 + s, ...`. Features per token are `[y, Y, j, J]` (DecPFN-no-St `[j, J]`),
  `y` and `j` summed since the previous token. `Y` and `y` are rebuilt from the `Trial`: Boiler
  `[temperature, u]` with the heater command recovered from the stage costs, Three-Tank
  `[V2, V3, V4]`, PT2 `[y]`, each with its per-step increments, Cartpole
  `Y = [x - x_ref, phi, action]` and `y = [xdot, phidot, Δaction]` with the action recomputed from
  `θ` and the previous measurements by the bang-bang rule, Furuta
  `[phi_arm, phi_pend, phidot_arm, phidot_pend, 0]` with per-step increments.
- `LegacyLCPFNPredictor` feeds the normalized cost at the old grid points
  `linspace(0, T - 1, L).astype(int)` that lie before `t_obs` and predicts the remaining ones.
  `predict` returns the horizon, `predict_curve` every remaining grid point (1-based steps).

Parity with the old code (`tests/unit/test_legacy_single_trajectory.py`, fixtures from the old
environment on golden rollouts of `tests/golden`, nominal plant, noise off):

| fixture | old inputs through the legacy network | new simulator through the legacy predictor |
|---|---|---|
| DecPFN Cartpole binned 1M | 7e-9 | 9e-8 |
| DecPFN-no-St Cartpole binned 1M | 3e-9 | 4e-7 |
| DecPFN Furuta binned 1M | 2e-9 | 1e-7 |
| DecPFN Boiler, Three-Tank, PT2 binned 1M | ≤ 1e-8 | ≤ 4e-6 |
| DecPFN Cartpole Gaussian 1M | relative 5e-9 | relative 9e-7 |
| LC-PFN Cartpole binned 1M (linear input norm) | 1e-9 | 2e-6 |
| LC-PFN Cartpole Gaussian 1M | relative 5e-8 | relative 1e-7 |
| LC-PFN Furuta binned 1M | 2e-9 | 2e-9 |

The numbers are the largest absolute deviation of the bar masses and `p_crash` (relative
deviation of mean and variance for the Gaussian heads). The old features of all five problems are
reproduced from the new trajectories to a relative 1.4e-6. The tests that need the model weights
skip when the export directory is absent.

## 4. Differences to the models of this package

The legacy models follow the old code. The DecPFN and LC-PFN of this package follow the thesis
text, with the remaining choices listed in [DEVIATIONS.md](DEVIATIONS.md).

| item | legacy (old code) | this package | reference |
|---|---|---|---|
| DecPFN covariates | `[y, Y, j, J]`, `Y` includes extra channels (Boiler heater command, Cartpole action and goal-relative position, Furuta constant target), Furuta treats the velocities as levels with summed increments | `[ΔJ, ξ, Σξ̇]` plus `J` on the thesis channels | [DEVIATIONS.md](DEVIATIONS.md#problems-and-priors-datamd-srcesbopfnproblemsreadmemd) "State channels" |
| DecPFN tokens | 0-based steps `0, s, 2s, ... < T - 1`, the first token covers one step, `N_grid` tokens (Cartpole 67) | `t_l = l·s`, `l = 1..N_grid − 1` | [DEVIATIONS.md](DEVIATIONS.md#problems-and-priors-datamd-srcesbopfnproblemsreadmemd) "Grids" |
| Boiler stride | 10 (71 grid points) | 11 | same |
| Time input | standardized token index `linspace(−√3, √3, L − 1)` (DecPFN), `index/(L − 1)` standardized as uniform (LC-PFN) | `t` z-scored over its range | [TRAINING.md](TRAINING.md#calibjson) |
| LC-PFN grid | `linspace(0, T − 1, L)`, `N_grid + 1` points including the first step | `t_l = ⌊l T / N⌋`, `l = 1..N` | [DEVIATIONS.md](DEVIATIONS.md#problems-and-priors-datamd-srcesbopfnproblemsreadmemd) "Grids" |
| LC-PFN cost input | clipped to `[0, cap]`, mapped to `[0.02, 1]`, then the equal-width CDF of the calibration values (`-linearnorm`) or a z-score (`_zscore` leaves) | z-score from prior samples | [DEVIATIONS.md](DEVIATIONS.md#predictors-and-training-trainingmd-cross_trajectorymd) |
| LC-PFN crashed rows | kept in the loss (trained before crash masking existed, `mask_crashed` absent) | removed from the batch | [TRAINING.md](TRAINING.md#the-thesis-recipe) |
| Training size | 10 or 100 epochs of 10,240 rows (102,400 or 1,024,000), 800 or 8,000 steps | exactly 100,000 or 1,000,000 rows | [DEVIATIONS.md](DEVIATIONS.md#deviations-from-the-thesis-text) D3 |
| Schedule and precision | DecPFN warmup from 0.01·lr and cosine to 1e-6 per step, LC-PFN per-epoch steps and fp16 autocast | warmup from 0, cosine to 0 per step, bf16 | [TRAINING.md](TRAINING.md#the-thesis-recipe) |
| Statistics | one 10,000-row batch of the training pool | 20,000 rows, up to 100,000 horizon values | [DEVIATIONS.md](DEVIATIONS.md#predictors-and-training-trainingmd-cross_trajectorymd) "Calibration sample sizes" |
| Seeds | none (time-seeded pools, unseeded initialization) | every draw seeded | [TRAINING.md](TRAINING.md) |
| Prior | old priors (single-channel problems without the per-channel jitter, rejection-truncated `p̂`) | thesis prior | [DEVIATIONS.md](DEVIATIONS.md#problems-and-priors-datamd-srcesbopfnproblemsreadmemd) "Sensor noise", "Parameter estimate" |
