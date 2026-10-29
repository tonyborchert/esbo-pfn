# Predictors

Every predictor maps a query trial (a controller observed for `t_obs` steps, `t_obs = 0` for an
unrun controller) and a context of other trials to a distribution over its total cost `J_T`.
This page covers the shared interface and the baselines (Segment-GP, Trajectory-TabPFN,
Segment-TabPFN). The trained models are documented in [TRAINING.md](TRAINING.md) (LC-PFN, DecPFN) and
[CROSS_TRAJECTORY.md](CROSS_TRAJECTORY.md) (XDecPFN, FT-PFN).

## Interface (`esbopfn.predictors.base`)

### Prediction types

Both types describe a mixed distribution with a crash atom at `+inf` and store unconditional
masses, so the value part integrates to `1 - p_crash`.

| Type | Value part | Produced by |
|---|---|---|
| `BarPrediction(edges, probs, p_crash)` | bars on strictly increasing `edges` (raw cost units), uniform density inside a bar, bars `(a, b]` with the lowest edge in the first bar | bar heads, Trajectory-TabPFN |
| `GaussianPrediction(mean, var, p_crash, lower=-inf)` | `N(mean, var)` truncated to `[lower, inf)`, `var = 0` is a point mass | Gaussian heads, Segment-GP, Segment-TabPFN |

Methods shared by both:

- `log_density(y)`: log of the unconditional value density, `log((1 - p_crash) f(y))`.
  `log_value_density(y)` is `log f(y)`.
- `log_prob_crash()`, `cdf(y)` and `survival(y)` of the mixed distribution (the crash atom is
  never below a finite `y`), `value_survival(y) = P(J > y | no crash)`.
- `quantile(q)` of the mixed distribution, `+inf` above `1 - p_crash`.
- `sample(rng, n)`: `+inf` for crash draws.
- `mean_var()`: mean and variance of the value part given no crash. For bars this includes the
  within-bar variance `width^2 / 12`.
- `conditioned_above(j)`: renormalizes the value mass above `j` to `1 - p_crash` and keeps
  `p_crash`. This is the conditioning used by the metric and by sampled virtual observations.
- `truncate_below(j)`: conditions the whole mixed distribution on `J >= j`. The crash atom lies in
  that event, so `p_crash` becomes `p / (p + (1 - p) S(j))`.

`Prediction = BarPrediction | GaussianPrediction`.

### Conditioned log-likelihood

`conditioned_log_likelihood(pred, y_true, j_obs, floor=1e-10)` is the metric of the prediction
benchmark. For a crashed trial (`y_true = inf`) it is `log p_crash`. Otherwise it is

    log(1 - p_crash) + log f(y_true) - log S_f(j_obs)

where `S_f(j_obs)` is the value mass above the cumulative cost at the cutoff. The crash probability
keeps its full predicted mass. Each of the two terms is floored at `log(floor)`, and the
conditioning is skipped when `S_f(j_obs) < floor`. Without floor events the score equals
`pred.conditioned_above(j_obs).log_density(y_true)`. `conditioned_ll_details` also reports which
floors were hit.

### Predictor protocol and registry

```python
class Predictor(Protocol):
    name: str
    supports_unrun: bool      # accepts t_obs = 0
    context_aware: bool       # uses the context
    def predictable_steps(self, spec) -> np.ndarray          # 1-based cutoffs
    def predict(self, query: Trial, context: Sequence[Trial]) -> Prediction
    def predict_many(self, queries, context) -> list[Prediction]
```

`PredictorBase` supplies the defaults and a looping `predict_many`. Predictors are built from id
strings with `get_predictor(id, problem, **kwargs)`, where `problem` is a `ProblemSpec` or a problem
name:

| Id | Predictor |
|---|---|
| `segment-gp` | Segment-GP of ESBO-GP |
| `tabpfn/<L>` | Trajectory-TabPFN on `L` grid points |
| `segment-tabpfn/split`, `segment-tabpfn/rate` | Segment-TabPFN |

A family is the first id component. It is resolved in this order:

1. a factory registered with `register_predictor_family(prefix, factory)`, where `factory` is a
   callable `factory(args, spec, **kwargs)` (`args` are the remaining id components) or a
   `"module:attribute"` string imported on first use,
2. the built-in `LAZY_FAMILIES` table,
3. convention: the module `esbopfn.predictors.<prefix>` (dashes become underscores) exposing
   `make_predictor(args, spec, **kwargs)`.

The trained models therefore register themselves by providing
`esbopfn/predictors/decpfn/__init__.py` (and `lcpfn`, `xdecpfn`, `ftpfn`) with a
`make_predictor`, without touching the registry module.

## Exact GP (`esbopfn.predictors.gp`)

A port of the GP used by the original MATLAB ESBO code, in PyTorch (float64, CPU):

- constant mean, ARD squared-exponential kernel `sf^2 exp(-0.5 sum_i (x_i - x'_i)^2 / ell_i^2)`,
  homoscedastic Gaussian noise `sn^2`, optional known per-point noise variances,
- inputs mapped to the unit cube of the search box, outputs standardized (std floor `1e-10`),
- hyperparameters `log ell_1..d, log sf, log sn` in natural-log standard deviations (GPML layout),
- negative log marginal likelihood plus GPML's smooth-box length-scale prior (`eta = 150000`) when
  no per-point noise is known, minimized with Adam:

| Setting | Value |
|---|---|
| Adam steps per cold fit | 100 |
| learning rate | 0.1 |
| gradient-norm clip | 100 |
| length-scale bounds | `[0.01, 0.50] / sqrt(2 ln 10)` (correlation 0.1 reached within 1 % to 50 % of the box) |
| signal std bounds `sf` | `[e^-4, e^1]` |
| noise std bounds `sn` | `[e^-10, e^3]` |
| initialization | `log ell` at the midpoint of its log bounds, `log sf = 0`, `sn = 0.1`, mean 0 |
| Cholesky jitter ladder | `1e-10, 1e-8, 1e-6, 1e-4` |

Hyperparameters are clamped into their bounds after every step. The fit draws no random numbers.

`cold_refit_growth` is a documented deviation (D2 in [DEVIATIONS.md](DEVIATIONS.md)). An `ExactGP` instance
is one chain of fits. A fit is cold (fixed initialization, 100 steps) unless a previous fit exists
and the training set has fewer than `(1 + cold_refit_growth)` times the rows of the last cold fit.
Then it starts from the previous hyperparameters and runs 15 % of the steps. `0.0` (the default)
makes every fit cold. Optimization runs use `0.05`.

Acquisition helpers: `mes_acquisition(gp, candidates, rng, n_samples=20, grid=512, incumbent)`
is max-value entropy search for minimization. It samples 20 minima `y*` by inverting the
distribution of the minimum over the candidates on a 512-point grid, caps them at the incumbent and
scores `mean_k[gamma phi(gamma) / (2 Phi(gamma)) - log Phi(gamma)]` with `gamma = (mu - y*) / sigma`.
`sample_positive(mean, std, lower, rng)` draws from the normal truncated to `[lower, inf)` by
inverse CDF.

## Segment-GP (`segment-gp`)

The section GPs of ESBO-GP. The segment boundaries are the sorted unique termination times of the
non-crashed context trials together with the horizon `T`. One exact GP per segment `(a, b]` is fit
on the segment cost `J_b - J_a` of every non-crashed context trial that observed it completely.
Crashed context trials contribute neither boundaries nor data. A query cut at `t_obs` gets `t_obs`
as its leading boundary, an unrun query starts at 0. The prediction is

    GaussianPrediction(J_{t_obs} + sum_q max(mu_q, 0), sum_q sigma_q^2, p_crash = 0)

with `sigma_q^2` including each segment GP's learned noise. A segment table with fewer than two
rows or a constant target uses the mean and variance of its data, and a segment that no
non-crashed trial observed completely is skipped. `predict_many` (also `predict_finals`) fits the
shared segments once for all queries. `cold_refit_growth` is passed to every segment GP, whose
chains are keyed by the segment. Predictions are defined at every cutoff.

## TabPFN baselines

### Checkpoint

The predictors use the released TabPFN v3 regressor `tabpfn-v3-regressor-v3_default.ckpt`
(Hugging Face `Prior-Labs/tabpfn_3`, about 233 MB) with the `tabpfn` package 9.x. The file is
loaded by absolute path from `$PFN_HOME/checkpoints/tabpfn/`. TabPFN's own cache directory and its
default model version (v3.5 in 9.x) are never used. If the file is absent, building a TabPFN
predictor raises `CheckpointMissing(path, instructions)`.

```
pfn checkpoints status [--verify]          # --verify compares the SHA-256 with the thesis file
pfn checkpoints download tabpfn            # TabPFN's downloader (license acceptance, see below)
pfn checkpoints download tabpfn --from F   # copy a local copy of the checkpoint
```

The v3 weights are gated. The downloader needs a one-time license acceptance, either through the
browser login TabPFN opens or through an API key from https://ux.priorlabs.ai exported as
`TABPFN_TOKEN` (accept the license on the Licenses tab first). Access to the Hugging Face repository
may also require `HF_TOKEN`. On a cluster without a browser, download the file once elsewhere and
use `--from`.

Inference uses `n_estimators = 8`, `inference_precision = "auto"`, all other TabPFN parameters at
their defaults, `ignore_pretraining_limits = True` (contexts can exceed 1000 rows on CPU), device
`mps`, else `cuda`, else `cpu`, and queries in chunks of 64 rows.

### Trajectory-TabPFN (`tabpfn/<L>`)

Grid `curve_grid(spec, L)`, `t_l = floor(l T / L)` for `l = 1..L`. Every non-crashed context trial
`k` contributes rows `[t_l, theta_k] -> J_{t_l}(theta_k)` (raw cumulative cost) for all grid points
`t_l <= T_k`. A running query that is not part of the context adds its own observed prefix. One
TabPFN fit answers all queries of a `predict_many` call at `[T, theta_i]`. The output is a
`BarPrediction` on TabPFN's 5000 bars (softmax of the `"full"` logits over `criterion.borders` in raw
units, the half-normal tails folded into the outer bars) with `p_crash = 0`. A complete query
returns its observed cost as a point mass, and a context with fewer than two rows or a constant
target returns the observed cost of each query. The thesis evaluates `L in {25, 50, 100, 250}`, and
ESBO-T-TabPFN uses `L = 50`.

### Segment-TabPFN (`segment-tabpfn/split`, `segment-tabpfn/rate`)

Same segment layout as Segment-GP.

- `split`: one TabPFN per segment on `theta -> segment cost`, queried at each query's `theta`.
- `rate`: one TabPFN on `[theta, q] -> segment cost / segment length` over all segments, where
  `q` is the index of the segment's end among the boundaries, queried at `[theta_i, q]` and scaled
  back by the segment length.

Each segment's bar distribution is reduced to its mean (clamped at 0) and variance, and the total
is `GaussianPrediction(J_{t_obs} + sum mean_q, sum var_q, p_crash = 0)`. Degenerate segment tables
use the mean and variance of their data.
