# Cross-trajectory predictors: XDecPFN and FT-PFN

XDecPFN and FT-PFN predict the total cost `J_T` of a trial (with a crash probability) from its
observed prefix and from the other trials of the same run. Both are trained once per problem on
the trajectory-set prior and are then used without further training, as context-aware predictors
for the prediction benchmark and as surrogates of the pure model loops (XDecPFN-PI/EI,
FT-PFN-EI/PI). This page describes both models as implemented, the data they need, how to train
them and how the thesis checkpoints are imported.

Code: `esbopfn.predictors.xdecpfn` and `esbopfn.predictors.ftpfn`, each with `model.py`,
`data.py`, `train.py`, `predictor.py` and `legacy.py`. Hyperparameters:
`src/esbopfn/configs/hparams/{xdecpfn,ftpfn}.yaml`.

## Quick start

```bash
pfn data gen --for train xdecpfn --problem cartpole     # sets/cartpole/xdecpfn (+ held-out pools)
pfn data gen --for train ftpfn --problem cartpole       # contexts/cartpole/ftpfn (+ held-out)
pfn train xdecpfn --problem cartpole                    # -> xdecpfn/cartpole/default/default
pfn train ftpfn --problem cartpole                      # -> ftpfn/cartpole/default/default
```

From Python:

```python
from esbopfn.predictors import get_predictor
p = get_predictor("xdecpfn/cartpole/default/default", "cartpole", device="cuda")
pred = p.predict(running_trial, history)            # BarPrediction over raw J_T
preds = p.predict_unrun(thetas, history)            # one BarPrediction per unrun controller
```

An id without a trained model raises `ModelNotTrained` with the training command as hint.

## XDecPFN

Configuration (thesis table of the cross-trajectory configurations): `d_model 384`, 3 causal and
6 cross layers, 6 heads, `d_ff 768`, dropout 0, 11 M parameters, `N_bin 1000`.

**Stage 1.** A causal pre-LN decoder runs over the tokens of one trajectory on the DecPFN grid,
`t_l = l N_stride` for `l = 1..N_grid - 1` (`grids.decoder_grid`, 66 tokens on Cartpole). A
token is the linear embedding of the z-scored vector `[ln J_{t_l}, Delta J_l, xi_{t_l},
sum xi_dot, t_l]` with the DecPFN covariates of `grids.tokens` (the increment `Delta J` stays in
raw cost units, the cumulative cost enters as `ln J`). Channel statistics come from the first
2,000 training instances, the time is z-scored over `{1..T-1}`. Grid points at or after a crash are
not tokenized. The output at a trajectory's last observed token is its representation, so stage 1
runs once per trajectory and serves every cutoff.

**Stage 2.** Each trajectory becomes one token: its stage-1 representation plus `e_theta(theta)`
(a linear layer on the controller, z-scored with the moments of the uniform prior on the box) plus
`e_complete`, `e_cut` or `e_crash`. Every unrun query is `e_theta(theta*) + e_unrun`. Six
bidirectional pre-LN blocks run over the trajectory and query tokens of one example. A token
attends to all trajectory tokens and to itself, never to a query token. A final LayerNorm and one
linear layer produce `N_bin + 1` logits, the bars over `ln J_T` plus the crash atom. A trajectory
that crashes before the first grid point has no stage-1 token and uses the learned vector `empty`.

**Loss.** The crash-aware bar loss (`-log a_inf` for a crash, else `-log a_k + log width_k`) on
the bins of `ln J_T`, averaged over all predictions of the batch. The bin edges are the equal-mass
quantiles of the non-crashed horizon values of the first 2,000 training instances (pool
trajectories and label trials). Targets outside the edges fall into the outermost bins.

**Training batch** (thesis, XDecPFN training). One step takes `N_sys = 8` consecutive instances
of the set pool. Stage 1 runs once over their `8 N_max` trajectories. Each instance yields
`N_reuse = 256` examples, each drawing `n ~ U{1..N_max}` of the pool's trajectories without
replacement, one cut probability `p ~ U(0, 1)`, a cut for every trajectory with at least one grid
token with probability `p` at a token uniform over its grid tokens, and `N_unrun = 8` unrun
queries from the instance's 128 label trials. The cut trajectories (predicted at the horizon) and
the unrun queries enter the loss. The cross stage runs on the examples sorted by size in chunks of
256, padded to the chunk's longest example. Optimizer: AdamW, learning rate `1e-3`, weight decay
0, 25 % linear warmup and cosine decay to 0, bf16 autocast on CUDA, no gradient clipping, 16,550
steps (132,400 instances, one pass) on every problem.

**Inference.** A completed trial enters with all its tokens and `e_complete`, a crashed one with
its tokens before the crash and `e_crash`, an early-stopped one with its tokens up to the cutoff
and `e_cut`. The running trial is one more cut trajectory and its token gives the prediction. A
running trial without a grid token yet, and an unrun controller, are predicted through an unrun
query. `predict_many` scores many running trials or cutoffs against one context (stage 1 cached per
trajectory), `predict_unrun` scores many controllers in one pass (queries do not interact). The
prediction is a `BarPrediction` on the edges `exp(ln_edges)` with the same masses, read as
uniform in `J` inside a bar. `predictable_steps` is the decoder grid.

## FT-PFN

Configuration: `d_model 512`, 6 layers, 4 heads, `d_ff 1024`, dropout 0, 13 M parameters,
`N_bin 1000`, curve grid of `N_grid = 8` points `t_l = floor(l T / 8)`.

**Tokens** (thesis eq. ftpfn:context). A trial of length `T_i` (its crash step if it crashed,
else its observed length) contributes `((theta_i, t_l), J_{t_l})` for the grid steps
`t_l <= T_i` and `((theta_i, T_i), J_{T_i})` if `T_i < T`. The token embedding is the sum of
linear embeddings of the z-scored controller, time (over `{1..T}`) and raw cost (statistics from
the first 20,000 training contexts, values clamped into the range of those samples). A crash
(`J = inf`) replaces the value embedding by the learned `e_inf`. A query `(theta, T)` has no
value embedding. Queries are the horizon of every trial that is truncated (neither crashed nor
completed) plus `N_unrun = 16` unrun controllers.

**Encoder.** Pre-LN blocks, final LayerNorm, one linear layer with `N_bin + 1` logits (bars over
raw `J_T` plus the crash atom). A batch of 32 contexts is packed into rows of the longest
context's length (first-fit decreasing, a context never spans two rows). The attention mask is
the PFN block mask: a token attends to the context tokens of its own context and to itself
(`model.block_mask`, plain `scaled_dot_product_attention` with a boolean mask).

**Loss and training.** Crash-aware bar loss on raw `J`, averaged over the queries of each
context, then over the contexts. Edges from the total costs of the first 20,000 training
contexts. AdamW with weight decay 0 (Adam), learning rate `1e-4`, 25 % warmup and cosine decay,
bf16 autocast on CUDA, no gradient clipping, batch 32, 109,375 steps (3,500,000 contexts, one
pass).

**Inference.** Context trials are tokenized as observed, the running trial adds its observed
prefix as a truncated trial, the query is `(theta, T)`. `predict_many` packs one context per query,
`predict_unrun` puts many unrun queries into one context. `predictable_steps` are the curve grid
points before `T` (the pure loops evaluate stops on the decoder grid).

## Data

| Model | Training data | Validation | Generate |
|---|---|---|---|
| XDecPFN | `sets/<p>/xdecpfn`, 132,400 screened instances with `N_max` trajectories each on the decoder grid plus 128 label trials | `sets/<p>/xdecpfn_heldout`, 256 instances | `pfn data gen --for train xdecpfn --problem P` |
| FT-PFN | `contexts/<p>/ftpfn`, 3,500,000 pre-tokenized contexts | `contexts/<p>/ftpfn_heldout`, 2,000 contexts | `pfn data gen --for train ftpfn --problem P` |

The formats are described in `docs/DATA.md`. A pool shorter than the run is cycled (only for
tests and smoke runs).

## Training

`pfn train xdecpfn|ftpfn --problem P [--seed S] [--run R] [--steps N] [--time-limit S]
[--device auto|cuda|mps|cpu] [--hparams FILE]`. In Python,
`esbopfn.predictors.xdecpfn.train.train(problem, hparams_row=None, seed=0, run="0",
out_dir=None, steps=None, time_limit=None, device="auto", train_pool=..., val_pool=...)` and the
FT-PFN twin (`train_store`, `val_store`). `hparams_row` overrides entries of the hyperparameter
file (smaller widths for tests).

A run writes the model directory `<PFN_HOME>/models/<model>/<problem>/default/default/<run>/`:
`model.pt` (state dict and architecture), `calib.json` (bin edges, z-score statistics, grid),
`config.yaml` (resolved hyperparameters, datasets, seed, git hash) and `metrics.json` (loss curve,
held-out NLL of the horizon predictions `val_nll_horizon`, of the unrun queries `val_nll_unrun`
and of both `val_nll`, the FLOPs of one step `flops_per_step` and `flops_total`). The FLOPs are
measured with `torch.utils.flop_counter.FlopCounterMode` over the forward and backward pass of the
first batch, the thesis' totals are stored next to them (`flops_thesis`). On Cartpole at the
thesis configuration one step measures 1.29e13 (XDecPFN, 2.13e17 for 16,550 steps, thesis
2.06e17) and 2.1e12 to 2.3e12 (FT-PFN, 2.3e17 to 2.5e17 for 109,375 steps, thesis 1.95e17, the
count includes the padding of the packed rows).

## Thesis checkpoints

The XDecPFN and FT-PFN weights behind the thesis results were trained with an earlier
implementation. They are imported unchanged as `variant=thesis`
models and served by small legacy classes that reproduce the old forward pass
(`predictors/xdecpfn/legacy.py`, `predictors/ftpfn/legacy.py`, strict state-dict loading):

```python
from esbopfn.benchmarks.import_models import import_thesis
import_thesis("~/Downloads/dynformer_timing_bundle", "~/Downloads/c1r2_furuta_lnJ")
get_predictor("xdecpfn/furuta/thesis/default", "furuta")
get_predictor("ftpfn/boiler/thesis/default", "boiler")
```

`import_thesis(from_dir, *extra_dirs, home=None, problems=None, overwrite=False)` (the function
behind `pfn import thesis-models --from DIR`) looks for `[checkpoints/]fr2/<p>/final.pt`,
`[checkpoints/]c1r2_qloop/<p>/f_*_seg*_final.pt`, `[checkpoints/]c1r2_calib/qloop/calib_c1r2_<key>_lnJ.json`
and checkpoints or calibrations lying directly in a directory, keeps the finished checkpoint with
the most steps per model and problem and writes `xdecpfn/<p>/thesis/default` and
`ftpfn/<p>/thesis/default`. `import_xdecpfn_checkpoint(ckpt, calib)` and
`import_ftpfn_checkpoint(ckpt)` import single files. `config.yaml` records the source file, its
SHA-256 and the step count. The legacy XDecPFN maps this package's trajectories onto the old
channels (Cartpole `[x - x_ref, xdot, phi, phidot]`, Furuta `[phi_arm, phi_pend, phidot_arm,
phidot_pend]`, Boiler `[temperature, u]` with the heater command recovered from the stage costs).
Both legacy predictors agree with the old code on a test context (XDecPFN Furuta to 1e-17 in the
bar masses, FT-PFN Boiler, PT2 dead-time and Furuta to 1e-7).

In the timing bundle the XDecPFN checkpoints of PT2 dead-time, Three-Tank and Boiler are the
ln-J sweep cells at 22,338, 32,435 and 58,785 steps, not the 16,550-step models of the thesis
tables. The importer imports them with a note in `config.yaml`. The thesis models of these three
problems (cell `nmatched`, 16,550 steps) were copied from the cluster
(`$HPCWORK/c1r2/<p>_lnJ_W128_qw05_nmatched/ckpt_final/f_s20260931_seg*_final.pt`) and imported
with `pfn import thesis-models --from <copy> --overwrite`, the copy laid out as
`c1r2_qloop/<p>/` and `c1r2_calib/qloop/` (planning/10_cluster_data_import.md).
