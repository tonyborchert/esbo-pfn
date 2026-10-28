# 03 — dynformer codebase audit (for the clean thesis-reproduction repo)

Scope: `impl/dynformer/` (and what it pulls from `impl/prediction/`). Read-only audit, 2026-10-28.
Paths are relative to `impl/dynformer/` unless they start with `impl/`, `writing/`, `~` or `/`.
"Cluster" means RWTH HPC (`/hpcwork/dln18500` = `$HPCWORK`, `/work/dln18500` = `$WORK`,
repo checkout at `/rwthfs/rz/cluster/home/dln18500/bachelorarbeit/impl/`).

**Headline findings**
1. **The thesis XDecPFN is not the documented "accepted" c1r2 recipe.** It is the later
   `lnJ_W128_qw05` sweep (ln-J bars, W=128, q_weight 0.5), retrained to 16,550 steps on
   pt2/threetank/boiler as cell `nmatched`. The authoritative description is
   `xdecpfn_ftpfn_hpo.md` + `xdecpfn_ftpfn_hpo_new.md`. `experiments/c1r2/README.md`,
   `c1r2_loss.py`, `c1r2_calib.py` and `c1r2_final.py` docstrings describe the raw-J/W=14 recipe and
   are stale for the thesis model. **There is no script for the `nmatched` retraining** (see §2.6).
2. **The thesis FT-PFN is `experiments/fr2` Config A** (`fr2_train.py` argparse defaults, 3.5 M
   contexts). For Furuta it is the `furuta180sg` key under `/work`, not the default `furuta` key.
3. **No thesis checkpoint is in git.** Partial local copies exist in `~/Downloads` (§2.7, §3.6).
   The `nmatched` weights (pt2/threetank/boiler) exist only on the cluster.
4. Several thesis claims disagree with code: gradient clipping, AdamW, time normalisation, boiler
   grid stride, the timing GPU, the ln-J acquisition integral, and the furuta kick_voltage freeze.
   See the discrepancy tables in §2.8, §3.7, §4.9 and §5.10.
5. 11 of the 61 generated PDFs come from hwplay_* script versions that exist only on the cluster.
   5 of those 11 are in the thesis (§6, §8).

---

## 1. MODULE MAP

### 1.1 Top level

| Path | What | Verdict | Entry points kept |
|---|---|---|---|
| `experiments/` | ~70 experiment dirs, 40+ dead (see 1.2) | KEEP-PARTIAL | §1.2 |
| `optplay/` | closed-loop optimization benchmark (package `optplay/optplay/`, `slurm/`, `instances/`, `results/`) | KEEP-PARTIAL | `python -m optplay.run`, `python -m optplay.aggregate [--crashes]`, `optplay/optplay/{q_loop,q_acq,q_predictor,fr2_q,fr2_predictor,c1_predictor,field,problems,screen,model_specs}.py` |
| `predplay/` | prediction benchmark (package `predplay/predplay/`, `tasks/`, `results/`, `slurm/`) | KEEP-PARTIAL | `python -m predplay.{genfield,tasks,predict,timing,aggregate}` |
| `thesis_figures/` | every thesis figure from this codebase (`python -m thesis_figures.<mod>`) | KEEP | `scripts/build_figures.sh`, the modules in §8 |
| `landscape_visualization/` | real/prior cost landscapes and furuta model landscapes (pickle caches in `out/`, gitignored) | KEEP-PARTIAL | `plot_landscape_grid.py` (lib), `plot_landscape_context_grid.py --mode {priors_sheet,furuta_sheet}`, `slice_candidate.py`, `merge_picked_rows.py`, `extra_instance.py`, `run_*.slurm` |
| `results_timing_bundle/` | prediction-timing JSONs behind the thesis x axes (`predplay_timing.json`, `summary.json`, `optplay_qloop_timing.json`) | KEEP (data) | — |
| `scripts/` | `slurm_submit.sh` (budget-gated sbatch), `build_figures.sh` (figure build), `build_pdf.sh`, agent-loop tooling (`run_iterations.sh`, `run_chains.sh`, `new_note.sh`, `resume_worker.sh`, `jupyter_start.sh`) | KEEP-PARTIAL: `build_figures.sh`, optionally `slurm_submit.sh` | — |
| `decisions/` | phase-gate LaTeX/PDF documents for the user | DROP (archive). Keep `06_F_recipe_ftpfn.tex` as provenance for FT-PFN | — |
| `notes/`, `library/`, `logs/`, `state/` | research log, papers, logs, agent bookkeeping | DROP | — |
| `xdecpfn_ftpfn_hpo.md`, `xdecpfn_ftpfn_hpo_new.md` | **the only correct description of both thesis models** (recipes, per-problem checkpoints, FLOP matching) | KEEP as docs (rewrite into README) | — |
| `README.md`, `CLAUDE.md`, `STATE.md`, `BACKLOG.md`, `SPEC.md`, `PROTOCOL.md`, `GRAPHSINEED*.md` | agent-workflow docs. `STATE.md` is frozen at 2026-09-10 and predates `nmatched` | DROP | — |
| `ENVIRONMENT.md` | HPC facts (§9) | KEEP-PARTIAL (condense) | — |
| `*.png`, `image*.png`, `texput.log`, `03_c5c6_architecture.log` | stray files | DROP | — |

### 1.2 `experiments/*`

**Final and relied on by the thesis**

| Dir | Role | Verdict |
|---|---|---|
| `c1r2/` | XDecPFN: model wrapper, tranche generator, calibration, loader, loss, trainer, lane/segment driver, held-out scorer, Ax sweep, arch grid, lr ladder, final runner. `driver/*lnj_w128_qw05*.sh` = the thesis training commands | KEEP-PARTIAL (§2) |
| `i5/` | `i5_q_arm.py` (`NoInj1QModel`, the XDecPFN class), `i5_arms.py` (`NoInj1Model`) | KEEP the 2 files |
| `s12/` | `s12_model.py` (`C1Config`, `C1Model`, `Block`, the base of XDecPFN), `GENERAL_LAW.md` | KEEP `s12_model.py` |
| `f3/` | `f3_model.py`: `nll_loss`, `build_bins`, `bin_index` (shared bar+atom loss). `f3_data.py` is used only by dead paths | KEEP `f3_model.py` |
| `s11/` | `s11_stream.py` (`CtxStreamZ`, the cartpole generator wrapper). Imported by `c1r2_gen.py:160` | KEEP `s11_stream.py` |
| `f2/` | `f2_stream.py` (`CtxStream`, `_build_context_v2`, the C++ generator wire format). Imported by `fr2_gen.py`, `fr2_check.py` | KEEP `f2_stream.py` |
| `plan05/` | `cap.py` (R1 trial cap `N_max = 3·15·d`). Imported by `c1r2_problems.py`, `fr2_env.py` | KEEP |
| `recipe/driver/` | `families.py`, `gpus.py`, imported by `c1r2_lane.py:56-57` (segment naming, busy-GPU parsing) | KEEP-PARTIAL or inline. Rest of `recipe/` (FLOP-matched "recipe 1", retired 2026-09-15) DROP |
| `fr2/` | FT-PFN: data generator, flat format, calibration, loader, model, trainer, drivers (§3) | KEEP |
| `oraclebench/` | `ob_gen.py`: Oracle-BO reference fields for predplay (genfield subprocess). Patches `experiments/f25/f25_genbench.py` textually | KEEP-PARTIAL (`ob_gen.py`), rewrite as a proper generator |
| `f25/` | `f25_genbench.py`, the field generator patched by `ob_gen.py` | KEEP until rewritten |
| `ladder/` | `ladder_predict.py`, `submit_ladder.sh`, `ladder_predict.slurm`, `genfield_181.slurm`, `results/` (local) for `context_ladder` | KEEP |
| `hwpred/` | `hwpred_ladder.py`, `hwpred_timing.py`, `hwpred_ladder.slurm`, `results/` (local) for the hwpred_* figures | KEEP |
| `timing/` | `cells*.txt`, `run_timing.slurm`, `run_crashagg.slurm` for `optplay_timing` and the crash pass | KEEP |

**Not in the thesis, but the landscape prior work** (`03_prior/*` notes): `landscape_sampler/`,
`landscape_probe/`, `ziegler_match/` (243 files). No thesis figure reads them (the prior figures
come from `landscape_visualization/`). → DROP (archive).

**Dead** (superseded tracks, one-off diagnostics, or retired recipes): `b1_adapters b38 b39 b59
boiler boiler_port f1 f4 f5 f6 f7 f8 f10 f11 f12 f13 f14 f16 f17 f18 f19 f20 f21 f22 f23 f24 f26
f27 f30 f31 f37 f39 fur128 fur225 i1 i2 i2b i4 i10h i11a i11b i11c playground predplay_decpfn_gap
pt64 pt135 s09 s13 s15 s50 s51 s7 threetank recipe(except driver/families.py,gpus.py)` and
the top-level `B*/F*/L*/S*_*.py` and `adhoc_*.py` scripts.

Notes on the dead list:
- `s50`/`s51` are Track-S loop studies on the old raw-J `QC`/`Q` cells. They are not thesis cells.
- `pt135` is only imported by the `s12` tests.
- `predplay_decpfn_gap/` holds diagnostic tables that may be mined for the thesis discussion, but no figure reads them.

**Import web caveat.** The XDecPFN code reaches across 8 dirs via `sys.path.insert`
(`c1r2_model.py:32-35`, `i5_arms.py:60-61`, `s12_model.py:36`). A clean repo should flatten it to one package.

---

## 2. XDecPFN (Track S, `dynformer_c1r2`, optplay `QC`/`QCn`)

### 2.1 Which model the thesis means
- The thesis model is cell `nmatched` on pt2_deadtime, threetank_pi and boiler. On cartpole and furuta_lqr_rigreset it is cell `lnjw128`, which holds the same weights there.
  - Resolver: `optplay/optplay/field.py:461-485` (`c1r2_xdecpfn_cell`, `c1r2_xdecpfn_spec`).
  - Tables: `C1R2_LNJ_ROWS` (field.py:~310-327), `C1R2_NMATCHED_ROWS` (field.py:408-418).
- predplay id: `dynformer_c1r2/cellnmatched`. cartpole/furuta rows are relabelled from `celllnjw128` by `predplay/predplay/aggregate.py:142-180`.
- optplay cell: `QCn PI abs 0.99` on pt2/threetank/boiler, `QC PI abs 0.99` on cartpole/furuta (`thesis_figures/style.py:1503-1520` prefix alias).
- Hardware (`qc_pi_0.99`): the furuta seg6 checkpoint (identical under both names).
- **NOT the thesis model:** the S-53/S-54 "accepted" c1r2 cell. It is `f_s20260931` under `$HPCWORK/c1r2/<p>/ckpt_final/`, raw J, W=14, and it is what `model_specs.c1r2_require()` resolves by default. Both share the name `f_s20260931`.

### 2.2 Architecture (exact)

Classes and arm:
- Class chain: `NoInj1QModel` (`experiments/i5/i5_q_arm.py:76`) → `NoInj1Model` (`i5_arms.py:76`, deletes θ-injection 1) → `C1Model` (`s12_model.py:98`).
- Built by `c1r2_model.build_from_cfg` with `v_feats=1`, so no extra lnJ channel. The `LnJQModel` (v_feats=2) at c1r2_model.py:103 is the unused "lnx" enrichment variant.
- Arm `noinj1q`.

Size and blocks:

| field | value | source |
|---|---|---|
| `d_model` | 384 | driver `--d-model 384` |
| heads | 6 (head_dim 64) | `c1r2_env.py:79` |
| layers | 3 causal within-trajectory + 6 bidirectional cross (driver `--layers-cross 6`; `c1r2_env.py:82` default 3 is the sweep value) | `experiments/c1r2/driver/*_lnj_w128_qw05.sh` |
| FFN | 2·d = 768, GELU (exact). `_mlp` | `s12_model.py:60-62` |
| block | pre-LN (`n1`→attn→residual, `n2`→FFN→residual), SDPA | `s12_model.py:67-95` |
| dropout | 0 (field exists, unused) | `s12_model.py:47` |
| params | 11,343,209–11,345,897 by problem (cartpole 11,345,897) | `experiments/c1r2/STATUS_TRANSFER.md` table |

Stage 1 tokens and encoding:
- Stage-1 token = `Linear(2C+4 → d)` on `[Z(C), dZ(C), yn, dyn, step/1000, ln step]` (`s12_model.py:115-125`). θ-blind. No positional encoding.
- `Z` = raw physical channels, z-scored with pool statistics.
- `yn` = **ln J_t z-scored** with global calibration constants (`j_mean/j_std` computed in ln space when `units="ln_J"`, `c1r2_calib.py:96-103, 400-445`). No clip.

Stage 1 grid (c1r2 rule):
- Dec-PFN grid `arange(0,T-1,k)` minus index 0, plus one token at the row's last observed step: `nat` if uncrashed, `nat-1` if crashed (`c1r2/README.md` §4, `c1r2_env.tokens_of`).
- `k` per problem: cartpole 15, pt2 8, threetank 7, **boiler 10**, furuta 9 (`optplay/optplay/model_specs.py:724-725`).
- `T_tok` = 67/63/68/70/67 (:727-728).

Stage 2 (one column = one cutoff configuration):
- Each context row is gathered at its cut token.
- Then add `θ-emb + marker`. The θ embedding is `th_mlp(th_ln(th_enc2(θ)))`, i.e. Linear→LayerNorm→2-layer MLP (`s12_model.py:150`). The marker is `nn.Embedding(4)`: cut / crashed / completed / unrun-query (`i5_q_arm.py:79-81`).
- `Q` query tokens per group = `θ-emb(θ*) + marker[3]`. No state, no time.

Stage 2 attention mask (`i5_q_arm.py:97-115`):
- Context attends only to its own group (block-diagonal over the column's 2 disjoint groups).
- A query attends to its group's context and to itself (`eye`).
- Nobody attends to queries.

Readout:
- `head(LN(x + skip))`, where skip = `h_gat` for context and the θ-embedding for queries (`s12_model.py:160`, `i5_q_arm.py:145`).
- `head = Linear(d, 1+1000)`: one softmax over `[crash atom, 1000 bars]`.

Output:
- Distribution of the final cost `J_T` (bars) plus crash probability (atom). No J_t curve.
- Bars are 1000 equal-mass quantile bins of **ln J_T** from non-crashed rows of the first 2000 tranche grids (`c1r2_calib.build_calib`, :317-331).
- At inference the edges are `exp`'d and mass is read as uniform-in-J within a bin (`optplay/optplay/c1_predictor.py:45-49, 485-496`).

### 2.3 Data pipeline (prior → tranche)

Generators (`c1r2_gen.py`, CPU), C++ binaries from `impl/prediction/tools` (`c1r2_problems.py:86-95`):
- `cartpole_cpp/cartpole_gen_s11`: cartpole, `cartpole/universal(1)`
- `plants_cpp/plant_gen_ctx2`: pt2_deadtime / threetank_pi / boiler `…/universal`
- `furuta_cpp/furuta_gen_ctx2` (+ `furuta_gen_phi` for φ-only screening): `furuta/lqr-new-realistic/rigreset(1,False)`
- Screen flags on every problem: `--screen-thetas 100 --screen-repeats 10 --screen-reliability 1.0 --screen-threshold 0.25` (`c1r2_problems.py:98-99`).

Per φ (one "grid" = one trajectory set, `c1r2/README.md` §5):
1. Draw φ (+φ̂, noise).
2. Pilot `N_PILOT=100` rollouts give `q_inst`.
3. Draw `s~U(0,q_inst)` and `n_crash~Bin(N_GRID,s)`.
4. Simulate `N_GRID` rows: 180/135/90/45/180 (`model_specs.py:734-735`). Furuta 180 is a user ruling and not 3·15·d=225 (`c1r2_problems.py:211-225, 397`).
5. Add `N_LABEL=128` extra (θ, J_T, crashed) label trials for the c=0 queries.

Furuta specifics:
- Rigs pre-screened by `c1r2_phi.py` (441,000 φ, ~233 core-h) and injected with `--phi-in`.
- Thesis tranche uses `--success-gate`: uncrashed rows re-stratified by `was_successful`. Data at `/work/dln18500/c1r2/furuta/{tranche,heldout}_success_gate`.

Load-time augmentation (`c1r2_data.py`), a pure function of `grid_seed(grid_idx, stream)` from module constant `SEED=20260931` (`c1r2_env.py:113`). It is **not** the CLI seed (`c1r2_final.py:26-48`).
- **W=128 columns per grid.**
- Per column the N rows are split into 2 disjoint groups, `n_a~U{1..N-1}`, i.e. 2 examples per column, so **N_reuse = 2·W = 256**.
- Censoring law `p_censor~U(0,1)` per (column, group), cut depth uniform in tokens ≥ `t_min=1` (`experiments/s12/GENERAL_LAW.md`).
- Q=8 c=0 queries drawn from the 128 label trials. The same Q θ serve both groups.

Volume and held-out:
- Tranche consumed single-pass in index order, **G=8 grids/step**. `nmatched`: 16,550 steps × 8 = **132,400 grids** on every problem.
- Tranche on disk is larger (cartpole extended to 800,000 grids).
- Held-out: 256 grids × 50 fixed columns, seed 20260932 (`c1r2_env.py:114-117`).

### 2.4 Training recipe (thesis cell)
- Training command: `uv run --no-sync --project ../prediction python experiments/c1r2/c1r2_final.py --d-model 384 --layers-within 3 --layers-cross 6 --lr 0.001 --W 128 --G 8 --Q 8 --q-weight 0.5 --b-cell 2.05669e+17 --seeds 20260931,20260932 --only f_s20260931 --tranche … --heldout … --calib …_lnJ.json --ckpt-dir … --out-dir … --lanes 3`, with `C1R2_PROBLEM=<key>`.
  - Per-problem copies live in `experiments/c1r2/driver/{cartpole,pt2,threetank_pi,boiler}_lnj_w128_qw05.sh` and `furuta_lnj_w128_qw05_successgate.sh`.
  - Launched over ssh on `login23-g-1` (login-node H100, GPU UUID `GPU-89dc5e46-…`), not via Slurm.
- Call chain:
  - `c1r2_final.py` → `c1r2_lane.Lane` (≤24-min segments, 27-min hard kill, resume from the newest `*_seg<k>.pt`) → `c1r2_grid.install_patches` → `c1r2_arch_train` → `c1r2_train.run` → `c1r2_loss.forward_loss`.
  - **6 files deep. Rewrite as one trainer.**
- Optimizer: `torch.optim.AdamW`, betas (0.9, 0.999), **weight_decay 0** in both groups (`c1r2_train.py:275-286`). No grad clipping (`c1r2_train.py:8`). Grad-norm spikes are only logged.
- Schedule: linear warm-up over `round(0.25·steps)` from 0.01·lr to lr, then cosine to 0, a function of the global step (`c1r2_train.py:150-170`). lr 1e-3.
- Loss: `cell_NLL / Σw_cell + q_weight·(query_NLL / Σw_q)` with q_weight 0.5 and atom_weight 1 (`c1r2_loss.py:200`).
  - Cell term = censored ("is_query") context cells, scored at the horizon.
  - Bars use log-density in ln-J units (`f3_model.nll_loss`). The atom uses log-mass.
  - Targets are winsorised into the support.
- Precision: bf16 autocast, loss in fp32.
- Budget: steps solved from a measured `FlopCounterMode` probe. For `lnjw128`: 2.05669e17 FLOP gives 16,550 / 22,338 / 32,435 / 58,785 / 16,550 steps (cartpole/pt2/threetank/boiler/furuta). `nmatched` = 16,550 everywhere.
- Seeds: 20260931 is the thesis cell. 20260932 is a second init seed of the same data order. **One training seed is used.**
- Cost: thesis says 32.1 min on cartpole (setup.tex Table `tab:setup:model-cost`).
  - B_cell = 25 min of an idle H100 at the S-53 anchor (`c1r2/STATUS.md:672-700`).
  - pt2 ran ~1 GPU-h at 22k steps.

### 2.5 Hyperparameter selection (what actually happened)
1. **S-53 Ax BO** (`c1r2_bo.py`, cartpole, raw J, arch d384 3+3):
   - Space: lr log[1e-5,1e-3], W int-log[8,256], G[8,32], Q[8,32]. 8 Sobol + qNEHVI, 20 cells, budget B_cell each.
   - Two objectives on the held-out set: prefix_nll and c0_nll, SE clustered by φ.
   - Selection = Pareto knee. Winner `t18`: lr 1e-3, W 14, G 8, Q 8 (`STATUS.md:26-40`).
2. **Architecture grid**, 8 FLOP-matched cells over d{384,512} × {3+3, 3+6, 6+3, 6+6}. Knee = **d384 3+6**, 11.35 M params (`STATUS.md:64-91`).
3. **lr ladder** {1.5e-3, 2e-3, 3e-3}: 1e-3 stands (`STATUS.md:106-145`).
4. **S-54 transfer** of that recipe to pt2/threetank/boiler (raw J, W 14), `STATUS_TRANSFER.md`.
5. **Hand-made follow-ups, 2026-09-12/13** (user requests, no search): ln-J calibration axis, then W 14→128 with q_weight 0.5 on pt2, then copied verbatim to all problems (`xdecpfn_ftpfn_hpo.md` §1.3). A `--grad-clip 1.0` A/B on pt2 was worse and reverted (§1.6). Furuta success-gate re-run.
6. **2026-09-20 `nmatched`**: the step count fixed at 16,550 (`xdecpfn_ftpfn_hpo_new.md` §4).

The thesis (setup.tex:153-158) says the cross-trajectory configs were selected "from smaller-scale tests, the search above and Rakotoarison's choices". That matches loosely. Steps 1-3 are a real HPO on cartpole that the thesis does not describe.

### 2.6 `nmatched` provenance gap
- `xdecpfn_ftpfn_hpo_new.md:130` says the three runs "resumed from an existing intermediate checkpoint of the §1 lane".
- No driver script, no `--steps`/`--max-steps` flag (`c1r2_final.py:1260-1318`), and no log is in the repo. `grep nmatched` hits only wiring and figures.
- Open questions:
  - What LR schedule did these checkpoints follow? A cosine planned for 22,338/32,435/58,785 steps, truncated at 16,550, would end at a non-zero lr. Retrained or rescheduled is unknown.
  - Do the segment indices seg3/seg2/seg2 mean resumption?
- Must be answered from the cluster logs (`experiments/c1r2/out/logs/*nmatched*`, `out/*_nmatched/`) or the checkpoint `step`/`lr` fields before the clean repo can claim reproducibility.

### 2.7 Inference API and checkpoints
- Predictive (running trial + history):
  - `optplay.c1_predictor.C1Predictor(problem, spec=field.c1r2_xdecpfn_spec(problem), max_context=spec.n_max).predict(partial, trials, [], predict_only_last=True)` returns `HistogramPrediction` (cost-unit edges) + `failure_prob`.
  - Context column representations are cached (`_ctx_cache`).
  - predplay uses `batch_policy="causal_prefix"` (`predplay/predplay/registry.py:538-541`).
- c=0 surrogate: `optplay.q_predictor.QPredictor(problem, spec=…).query_c0(trials, thetas)` (`q_predictor.py:186`), chunk 128 (`OPTPLAY_Q_CHUNK`).
- Loader `c1r2_model.load(ckpt, device)` (`c1r2_model.py:266-274`):
  - Rebuilds the config from `ck["mcfg"]`.
  - Requires `$C1R2_PROBLEM` exported before Python starts, because `c1r2_env` binds the problem at import and ~28 modules capture it. One problem per process.
  - The calibration basename is checked against the checkpoint.
  - Edges sha256 are pinned in `field.py:335-341` (`OPTPLAY_C1R2_EDGES_SHA256_<P>`).
- Checkpoint naming: `<ckpt-dir>/f_s20260931_seg<k>_final.pt`. Only the highest-numbered `done=True` segment is the trained model; earlier segments are partial.

| problem | thesis checkpoint (cluster) | calibration | local copy |
|---|---|---|---|
| cartpole | `/hpcwork/dln18500/c1r2/cartpole_lnJ_W128_qw05/ckpt_final/f_s20260931_seg2_final.pt` | `experiments/c1r2/out/calib_c1r2_cartpole_lnJ.json` | `~/Downloads/dynformer_timing_bundle/checkpoints/c1r2_qloop/cartpole/f_s20260931_seg1_final.pt` (renamed "seg1"; Sep 15 bundle = lnjw128) |
| pt2_deadtime | `/hpcwork/dln18500/c1r2/pt2_deadtime_lnJ_W128_qw05_nmatched/ckpt_final/f_s20260931_seg3_final.pt` | `…/out/pt2_deadtime/calib_c1r2_pt2_deadtime_lnJ.json` | **none** (bundle has only lnjw128 at 22,338 steps) |
| threetank_pi | `…/threetank_pi_lnJ_W128_qw05_nmatched/ckpt_final/f_s20260931_seg2_final.pt` | `…/out/threetank_pi/calib_c1r2_threetank_pi_lnJ.json` | **none** |
| boiler | `…/boiler_lnJ_W128_qw05_nmatched/ckpt_final/f_s20260931_seg2_final.pt` | `…/out/boiler/calib_c1r2_boiler_lnJ.json` | **none** |
| furuta_lqr_rigreset | `/work/dln18500/c1r2/furuta_lnJ_W128_qw05_successgate/ckpt_final/f_s20260931_seg6_final.pt` | `…/out/calib_c1r2_furuta_lnJ.json` | `~/Downloads/c1r2_furuta_lnJ/f_s20260931_seg6_final.pt` (md5 d9527b94…, identical to the bundle's `c1r2_qloop/furuta_lqr_rigreset`) |

Notes on local files:
- `experiments/c1r2/out/` is **not present locally**, so no calibration file is in the repo.
- Local lnJ calibrations: `~/Downloads/dynformer_timing_bundle/checkpoints/c1r2_calib/qloop/calib_c1r2_<p>_lnJ.json` (all 5) and `impl/prediction/c1r2_furuta_lnJ/calib_c1r2_furuta_lnJ.json`.
- The bundle's `c1r2_predplay/*` are the raw-J S-54 cells, **not thesis weights**.

### 2.8 Discrepancies, thesis vs code (XDecPFN)

| # | Thesis | Code | Comment |
|---|---|---|---|
| X1 | appendix.tex:350-352: AdamW, gradient norm clipped to 1, "as in both released implementations" | `c1r2_train.py:8, 275-286` AdamW wd 0, **no clipping** | clip tested and rejected (`xdecpfn_ftpfn_hpo.md` §1.6) |
| X2 | appendix.tex:334-338: time z-scored over {1..T−1} | `s12_model.py:115-125`: `step/1000` and `ln step`, not z-scored | |
| X3 | appendix.tex:330-331: continuous channels projected by one linear layer | θ embedding = Linear→LN→MLP (`s12_model.py:105-108, 150`) | |
| X4 | method.tex:1068-1072: first stage on "the same N_grid−1 tokens as DecPFN", final token = last grid point | c1r2 grid drops index 0 and adds a last-observed-step token (`c1r2/README.md` §4) | e.g. cartpole 66 grid + 1 = 67 tokens |
| X5 | appendix.tex Table `tab:app:grid`: Boiler N_stride 11, N_grid 64; "XDecPFN uses the same grid as DecPFN" (setup.tex:180) | boiler `k=10`, 69 grid tokens (`model_specs.py:724`, `c1r2_problems.py:193`, `predplay/predplay/recipe_group_tests.py:58`) | check which stride DecPFN's boiler checkpoint used |
| X6 | method.tex:1088: mask "restricts tokens from attending to query tokens" | also queries attend to themselves, not to other queries (`i5_q_arm.py:97-115`) | consistent in spirit |
| X7 | method.tex:1095-1098: final LN + linear head on the stage-2 output | readout adds a skip `x + h_gat` (context) / `x + θ-emb` (query) before LN (`s12_model.py:160`, `i5_q_arm.py:145`) | undocumented |
| X8 | method.tex:1124-1132: sample n ≤ N_max trajectories independently per example; mean of all losses | each column splits the pool into 2 **complementary** groups (`n_a~U{1..N-1}`). Loss = mean over cell queries + **0.5 ×** mean over unrun queries (`c1r2_loss.py:200`) | q_weight 0.5 is not in the thesis |
| X9 | setup.tex:192: "XDecPFN normalizes and bins ln J" | both input `yn` and bars in ln J. At inference the bars are exp'd and treated as uniform in J within a bin (`c1_predictor.py:45-49`). The pure-loop EI/PI integrate uniformly in **ln J** (`q_acq.py:72-89`), whereas method.tex:1149-1161 assumes uniform-in-J | predictive density used for LL ≠ the one used for acquisition |
| X10 | Table `tab:app:matched-flops` caption: "Measured total training FLOPs per problem" | the XDecPFN column equals 2.05669e17 × 16,550/steps_lnjw128 (5.79e16 = ×16550/58785, 1.52e17 = ×16550/22338) | derived, not measured |
| X11 | appendix.tex:409, setup.tex:119: H100 Slurm GPU jobs with 8 Xeon cores | trained on the login node `login23-g-1` H100 by ssh, under the 30-min kill, in segments | |
| X12 | setup.tex:194-197: 132,400 sets × N_reuse 256 | matches (G=8, W=128, 2 groups) | OK |
| X13 | Table `tab:app:ftpfn-xdecpfn-configs`: dropout 0, wd 0, d384, 3+6, 6 heads, 11 M, N_unrun 8, 16,550 steps | matches | OK |
| X14 | setup.tex:153: "selected individually from smaller-scale tests" | an Ax BO + arch grid + lr ladder exist (§2.5) | under-reported |

---

## 3. FT-PFN (Track F recipe 2, `dynformer_fr2`, optplay `QF`)

### 3.1 Definitive code
- `experiments/fr2/` README (format and law) and `decisions/06_F_recipe_ftpfn.tex` (§"Generation as built" items (g)-(k) for campaign 2, furuta, the log-J ablation and the predplay integration).
- Model `fr2_model.py`, trainer `fr2_train.py`, loader `fr2_loader.py`, calibration `fr2_calib.py`, generator `fr2_gen.py` (+ `fr2_gen.slurm`, `fr2_campaign2.sh`, `fr2_furuta180sg_gen.sh`), env table `fr2_env.py`, checks `fr2_check.py`, `fr2_train_check.py`, GPU driver `fr2_run_gpu0.sh`/`fr2_wait_gpu0.sh`.

### 3.2 Architecture (Config A = argparse defaults, `fr2_train.py:185-198`)
- `FR2Model`: d 512, 6 layers, 4 heads, FFN 1024 (GELU), pre-LN blocks, final LN, `head = Linear(d, 1+1000)` (`fr2_model.py:43-50, 111-149`).
  - Params 13,233,641–13,237,225 (furuta180sg 13,237,225).
- One packed flat sequence per batch of 32 contexts. Block PFN mask: same context AND (key is a context/null token OR self) (`fr2_model.py:91-103`). `flex_attention`, compiled on CUDA.
- Token = `E_t([t, ln t]) + E_θ(θ_z) + E_value(J~)` [+ cutoff_vec | crash_vec], with t = step/T.
  - `E_t` = MLP(2→128→d), `E_value` = MLP(1→64→d) on clipped, z-scored raw J, `E_θ` = Linear (`fr2_model.py:8-16, 139-165`).
  - One learned `null_ctx` per context.
- Grid: 8 points `round_half_up(T·j/8)`, j=1..8. Each trajectory contributes the grid steps < L plus a token at its last observed step L, with a marker cutoff/crash (crash at `nat−1`) (`fr2/README.md` "Tokenisation").
- Query = bare `E_t(1)+E_θ(θ)`, no value, no marker.
- Output: `[crash atom, 1000 bars]` on **raw J_T**.
  - Calibration: 1000 quantile edges over finite non-crashed query targets of the **held-out split** (`fr2_calib.py`, built once per run into `<out>/calib.json`).
  - Input clip low edge = union rule.

### 3.3 Data
- One φ per context, never reused. 3.5 M train + 2,000 held-out contexts per problem, fully materialised (`fr2-flat-v1`, `$HPCWORK/fr2_F/<problem>/{train,heldout}/`).
- Shard k seed = base + k, with base 261/262/263/264 M for cartpole/pt2/threetank/boiler (README "Sharding").
- Context law `_build_context_v2` (`experiments/f2/f2_stream.py`, `f2/GENERAL_LAW.md`):
  - N~U{1..n_max}, pool 3·n_max candidates, crash share `s~U(0,q_inst)`.
  - p_censor~U(0,1), at least one cut, cut uniform in [1, nat−1].
  - Queries: one horizon query per censored row + **16 held-out c=0 queries**.
- Furuta (`fr2_env.py` keys `furuta` D=5/N_MAX 225, `furuta180` D=4/180, **`furuta180sg` D=4/180 success-gated**):
  - Dataset `furuta/lqr-new-realistic/rigreset(1,False)`.
  - `furuta_cpp ctx` was extended with a per-instance `dead_time`/`kick` payload (header flag bit 3, binary `furuta_gen_ctx2`).
  - 32 Slurm shard jobs, ~1,790 core-h (decisions/06 item (i)).
  - Thesis checkpoint key: `furuta180sg`, data/runs under `FR2_WORK=/work/dln18500/fr2_F`.

### 3.4 Training
- Command: `uv run --no-sync --project ../prediction python experiments/fr2/fr2_train.py --problem <p> --contexts 3500000 [--out …]` (furuta: `--problem furuta180sg`, `FR2_WORK=/work/dln18500/fr2_F`).
  - Run via `fr2_run_gpu0.sh` on login23-g-1, relaunch-to-resume from `ckpt_last.pt` across the 30-min kill.
- Optimizer: `torch.optim.Adam` (not AdamW), lr 1e-4, eps 1e-8, wd 0, **no clipping** (`fr2_train.py:7, 283`). Non-finite grad norm skips the step.
- Schedule: 25 % linear warm-up, then cosine to 0 (`fr2_train.py:55-61`).
- Batch 32 contexts. steps = 109,375 = 3.5 M/32, single pass in file order. Seed 0. bf16 autocast.
- Loss = bar+atom NLL, mean over a context's own queries, then over the 32 contexts.
- Selection (decisions/06 item (g)-(h)):
  - Two configs FLOP-matched to the XDecPFN cartpole anchor 2.0567e17 (6ND + 12L²d count): A (d512, 3.5 M, 2.03e17) and B (d768/6×128 heads/FFN 1536, 29.3 M, 1.75 M ctx, 2.07e17).
  - A won on cartpole held-out NLL 4.383/4.698/3.472 vs B 4.398/… and the 1 M baseline 4.443/…, then ran on all problems. B was not run elsewhere.
  - The linear-vs-FFN head choice claimed in setup.tex:159-163 has no artefact in `experiments/fr2` (the F3 head is LN+Linear by construction). Its source is presumably earlier Track-F notes.
- **log J input ablation** (pt2_deadtime only):
  - `--value-space log1p` feeds ln(1+J) to `E_value` only. The head stays raw J (`fr2_train.py:185`, `fr2_calib.py`).
  - NLL all/horizon/c0: 2.028/2.161/1.736 vs raw 1.854/1.990/1.554.
  - Run dir `$HPCWORK/fr2_F/runs/pt2_deadtime_3500000_logj/`.
  - The thesis (setup.tex:189-192) words it as "applying a logarithm to J … opposite effect on FT-PFN". Correct in direction, but the ablation was log1p on the input only.

### 3.5 Inference API
- `optplay.fr2_predictor.FR2Predictor(problem).predict(partial, actual, …)` (`fr2_predictor.py:121-290`):
  - Past trials are tokenised uncensored. The running trial adds its observed prefix as a censored context row, then a bare query.
  - No caching. Every cutoff re-encodes.
  - Refuses `furuta_lqr`.
- `optplay.fr2_q.FR2QPredictor(problem, …).query_c0(trials, thetas)` for the pure loop (`fr2_q.py:143-376`).
  - xgrid knob: stop-rule refresh on XDecPFN's token grid (`fr2_q.py:181-194`).
- Checkpoint dir override `FR2_PREDPLAY_CKPT_DIR_<P>`. The furuta override `FR2_CELL_OVERRIDES` (`field.py:~347`) and `registry.py:601-603` are hardcoded to `/work/...furuta180sg_3500000`.

### 3.6 Checkpoints

| problem | cluster | local copy |
|---|---|---|
| cartpole | `$HPCWORK/fr2_F/runs/cartpole_3500000/final.pt` (+`calib.json`, `metrics.json`) | `~/Downloads/dynformer_timing_bundle/checkpoints/fr2/cartpole/final.pt` |
| pt2_deadtime | `$HPCWORK/fr2_F/runs/pt2_deadtime_3500000/final.pt` | bundle `fr2/pt2_deadtime/final.pt` |
| threetank_pi | `$HPCWORK/fr2_F/runs/threetank_pi_3500000/final.pt` | bundle |
| boiler | `$HPCWORK/fr2_F/runs/boiler_3500000/final.pt` | bundle |
| furuta_lqr_rigreset | `/work/dln18500/fr2_F/runs/furuta180sg_3500000/final.pt` | bundle `fr2/furuta_lqr_rigreset/final.pt` |

Verify that the bundle copies include each run's `calib.json`, which the predictor needs. Not checked here: no project Python was run.

### 3.7 Discrepancies (FT-PFN)

| # | Thesis | Code |
|---|---|---|
| F1 | appendix.tex:350-352: AdamW + clip 1 | Adam, no clip (`fr2_train.py:7, 283`) |
| F2 | appendix.tex:330-338: linear projections, t z-scored over {1..T} | t enters as `[t/T, ln(t/T)]` through a 2-layer MLP, value through a 2-layer MLP (`fr2_model.py:139-141`) |
| F3 | setup.tex:183-186: N_grid=8, ≤1,620 tokens on Cartpole/Furuta | matches (`n_max·8 + n_max + 16`: 1,636 incl. queries) |
| F4 | setup.tex:171-175: compute matched at ~2e17 on Cartpole, then config incl. step count fixed | matches. Note FT-PFN holds contexts fixed and XDecPFN (`nmatched`) holds steps fixed |
| F5 | setup.tex:189-192: log J "opposite effect on FT-PFN" | the ablation is input-only `log1p`, pt2 only |
| F6 | method.tex:1046-1050: horizon queries for trials that neither crash nor complete in context, plus N_unrun | matches (one per censored row + 16 held-out) |

---

## 4. PREDPLAY (prediction benchmark)

Root `predplay/predplay/`. Full detail is in the subagent audit, condensed here.

### 4.1 What it computes
- **Tasks** (`tasks.py`): context rungs `(0,2,4,8,16,32,64)` (:37).
  - Query = trial n, context = trials[:n] truncated at the Oracle-BO `t_obs`.
  - Target = hidden-continuation `J_T` or crash.
  - `selected` = first K=50 non-crashed queries per (problem, N), seed-major (:50, 116-124).
  - `crash_selected` = 50 crashed queries (none on boiler/threetank).
  - Frozen to `predplay/tasks/<p>.parquet` + `.json` with a sha16. `--verify` rebuilds.
- **Scoring** (`scoring.py`): wraps `benchmark.predictor_benchmark._score` from `impl/prediction`.
  - `ll` and `ll_conditioned`: value density truncated at J_obs and renormalised. A crash scores log p_fail.
  - Floor 1e-10. Flags `clamped`, `out_of_support`.
  - **No calibration metric exists.**
- **Cutoffs** (`predict.py:109-137`): `CUTOFF_FRACS=(0.1,0.3,0.5,0.7,0.9)` (`registry.py:66`), snapped down to the model's `predictable_timesteps`.
  - If none is ≤ the cutoff, it takes the first grid point after it. This bites FT-PFN at 10 % (→12.5 %).
  - Failure-channel models are additionally scored on their full own grid (for stop curves).
- **fig1** (`aggregate.py:280-309`):
  - `med_D` = median of `ll_conditioned` over all participating rows of a problem. `ll_norm = ll_conditioned − med_D`.
  - Cross y = median of `ll_norm` pooled over problems. Per-problem y = raw median.
  - Context-blind models are collapsed to `n_ctx=-1`.
- **fig2 / stop sheet** (`aggregate.py:313-440`): stop at the first step with `p_fail > τ`, τ over {0,1} ∪ 201 model quantiles.
  - y = mean steps saved over crashed queries, x = false-stop share of non-crashed queries.
  - "Steps saved at FSR ≤ 0.1" is read off in thesis_figures, not predplay.
- **fig3** (`fig3.py`): Pareto front of hand-picked series. thesis_figures imports `Series`, `_TARGETS`, `pareto_front`.
- **Context ladder**: separate protocol, `experiments/ladder/ladder_predict.py`.
  - 181-trial Oracle-BO field `$HPCWORK/predplay/field/ladder181_<p>`. Query fixed at trial 180, K=50.
  - Rungs pt2 134/67/33/16/8/4/2/1/0, cartpole/furuta 180/90/45/22/…
  - Plot = mean over cutoffs, then **mean** over queries (`thesis_figures/context_ladder.py:190-191`).

### 4.2 Registry (`registry.py`)
41 ids are registered. **The thesis 35** = decpfn 12 (`decpfn/epochs{10,100}/head{binned,gaussian,linear}[/states0]`) + lcpfn 6 + stenger 1 + tabpfn 12 (`resolution{25,50,100,250}/type{raw,ls,diff}`) + stengertpfn{rate,split} 2 + `dynformer_c1r2/cellnmatched` 1 + `dynformer_fr2` 1.

Excluded: `stengertpfn/modeminimal`, `dynformer_recipe/*` (retired), `lcpfn_zscore/*` (post-thesis, 2026-09-29).

Checkpoint resolution:
- DecPFN/LC-PFN: `impl/prediction/models/{decpfn,lcpfn}/<dataset>/transfer/<ep>/<head>[_nostates]/0/model.pt` via `manager.load.load_predictor`. **Absent locally.**
- TabPFN: `~/.cache/tabpfn/tabpfn-v3-regressor-v3_default.ckpt`.
- c1r2: `model_specs.c1r2_spec` plus `OPTPLAY_C1R2_CKPT_*`/`CALIB*`/`EDGES_SHA256*` env overrides, recovered from old logs (`predplay/FINDINGS.md:792-797`).
- fr2: `fr2_env.WORK/runs/<key>_3500000`, furuta hardcoded `/work/…`.
- Availability is a filesystem probe, so off-cluster ids silently vanish.

### 4.3 Problems, fields

| problem | seeds | runs | tasks | crashed tasks |
|---|---|---|---|---|
| boiler | 4 | 200 | 1400 | 0 |
| threetank_pi | 4 | 200 | 1400 | 0 |
| pt2_deadtime | 7 | 350 | 2450 | 858 |
| cartpole | 8 | 400 | 2800 | 839 |
| furuta_lqr_rigreset (thesis "Furuta") | 10 | 500 | 3500 | 839 |
| furuta_lqr (old plain variant, **drop**) | 12 | 600 | 4200 | 1042 |

- φ = optplay's screened `instances/<p>.json`, 50 per problem.
- Fields are generated by `genfield.py` → `experiments/oraclebench/ob_gen.py --trials 65` → patched `experiments/f25/f25_genbench.py` (Oracle-BO via `impl/prediction` `benchmark.OptimizerBenchmark`, `optimizers.oracle_bo`).
- Fields live at `$HPCWORK/predplay/field/<p>/run_phiNNNN_sMM.{npz,json}`, **cluster only**. Locally only the frozen task parquets exist.

### 4.4 Results storage
Local `predplay/results/`:

| file(s) | built | what it holds |
|---|---|---|
| `fig1_cross.csv`, `fig1_per_problem.csv`, `fig1_meta.json`, `fig2_{pt2_deadtime,cartpole,furuta_lqr,furuta_lqr_rigreset}.csv` | canonical, **2026-09-29** | **lcpfn replaced by lcpfn_zscore**, med_D changed |
| `*_pre_lcz.*` | before the swap | closest to the thesis. FT-PFN n=64 1.6182 vs thesis 1.618 |
| `*_<tag>.*` | | audit builds (DROP) |
| `results/timing/` | | not used by the thesis |

- `med_D`, canonical: boiler −4.6723, threetank 9.9019, pt2 −5.0832, cartpole −7.6728, furuta_rr −4.5805.
- `med_D`, pre_lcz: −4.6809, 9.8109, −5.3467, −7.6771, −4.5818.
- `results_facts.md` quotes the pre_lcz set, except furuta −4.5868, which matches neither.
- The thesis pooled numbers come from `writing/thesis/data/sources/fig1_cross_5prob_canonmed.csv`. It is a cluster-side hand recomputation that mixes 6-problem med_D rows (see `cluster_answers_2026-09-24.md` §6). **`aggregate.py` cannot reproduce it.**
- Raw predictions: `$HPCWORK/predplay/preds/<p>/<model>/part-*.parquet` + `shard-*.done.json`, cluster only. Per cluster_answers, `cellnmatched` rows are "no longer in the current prediction store".

### 4.5 Running
- From `predplay/`: `uv run --no-sync --project ../../prediction python -m predplay.{fields,genfield,tasks,predict,timing,status,aggregate,fig3}`.
  - `predict --problem P --group G --workers k --shard i/n`.
  - `aggregate --group … --cross-problem … --tag`.
- Env: `OPTPLAY_C1R2_CELLS=nmatched` must be exported for every stage. `C1R2_PROBLEM` per process.
- Slurm (`predplay/slurm/`): `predict.slurm`, `predict_gpu.slurm`, `timing*.slurm`, `genfield.slurm`, `c1r2_predplay.slurm`, `fr2_predict.slurm` (hardcoded `/rwthfs/...` paths). Accounts rwth2055 / thes2287.

### 4.6 Imports from `impl/prediction`
- `benchmark.predictor_benchmark`, `benchmark.glyph_comparison_plotter`, `manager.load.load_predictor` (→ `predictors/{decpfn,lcpfn,stenger,tpfn,stengertpfn}`).
- `model.system.Trial`, `model.predictor.GaussianPrediction`, `util/`.
- Through genfield: `benchmark.OptimizerBenchmark`, `optimizers.oracle_bo`, `util.matlab_gp`.

### 4.7 Thesis figures fed by predplay
- `fig3_sheet`, `fig3_per_problem` (fig3_cross + per problem), `tabpfn_panel`, `ftpfn_xdecpfn_panel`: read `fig1_*` + timing bundle.
- `fig2_sheet` / `ftpfn_xdecpfn_stop_sheet`: read `fig2_*`. The dashed noStates curve is recomputed from cluster parquets.
- `lcpfn_decpfn_heads_benchmark`, `lcpfn_decpfn_stop_sheet`: read cluster parquets.
- `context_ladder`: reads `experiments/ladder/results`.
- `prediction_example`: live inference.

### 4.8 Timing bundle (`results_timing_bundle/`)
- `predplay_timing.json`: 692 records, 37 ids, 5 problems, all N. `summary.json` = the same + 35 `optplay_qloop` records. `optplay_qloop_timing.json`.
- **Every record**: `host: artemis`, `device: cuda:Quadro RTX 6000`, `n_calls: 1`, `torch_threads: 1`, measured 2026-09-15.
- Produced by `~/Downloads/dynformer_timing_bundle/run_timing.py`, a self-contained harness with vendored code. It is not in the repo.
- It timed `dynformer_c1r2/cellf_s20260931`, i.e. **the raw-J S-54 c1r2 cells**: same architecture as the thesis XDecPFN, different weights. `thesis_figures/timing.py:88-93` (`ARCHITECTURE_TWIN`) maps them onto `cellnmatched`.
- `thesis_figures/timing.py:102-115` (`ESTIMATE_NONMONOTONE=True`) replaces 35 non-monotone (model, problem, N) cells with power-law extrapolations. 4 c1r2 records are hand-repaired (:168-197).
- The thesis quotes the cross-problem median of `seconds_median`.

### 4.9 Discrepancies (prediction benchmark)

| # | Thesis | Code |
|---|---|---|
| P1 | setup.tex:551-552: per-trial mean over the 5 cutoffs, then median | median over all (task × cutoff) rows directly (`aggregate.py:288-301`) |
| P2 | appendix.tex:411, setup.tex:561: Quadro RTX 4000 (8 GB) workstation | Quadro RTX 6000 on `artemis`, single call, partly extrapolated (§4.8) |
| P3 | setup.tex:551: "closest grid point before" the cutoff | falls back to the first point **after** (`predict.py:121-123`). FT-PFN 10 % → 12.5 % |
| P4 | setup.tex:570: stop when p_fail reaches τ, every τ∈(0,1) | `p_fail > τ` over 203 model-quantile thresholds |
| P5 | setup.tex:568-569: one crashed and one non-crashed group | crashed group = 50/cell for context models but the full field for context-blind ones. The ok group is the full field |
| P6 | setup.tex:558-559: "every participating score" for med_D | participants include non-thesis models (recipe arms, stengertpfn minimal, lcpfn_zscore in the canonical build) |
| P7 | setup.tex:530-532: instances sampled + screened | reuses optplay's 50 frozen instances with 4-12 seeds each |
| P8 | the context ladder protocol (181-trial field, fixed query, mean of means) | not described in setup.tex |

---

## 5. OPTPLAY (optimization benchmark)

Root `optplay/optplay/`. Every module imports `env.py` first, which puts `impl/prediction` and `impl/prediction/util` on sys.path and sets `OPTIM_DISABLE_USETEX=1`. `README.md` §1 and `field.py:43-58` docstrings are stale.

### 5.1 Instances and budgets
- Thesis problems: `boiler` (d=1, T=701), `threetank_pi` (2, 476), `pt2_deadtime` (3, 501), `cartpole` (4, 1000), `furuta_lqr_rigreset` (d=8, `budget_d=4`, T=600) (`problems.py:108-180`). Drop `furuta_lqr` and `furuta_pid`.
- Budgets (`problems.py:239-268`):
  - `trials_budget = 15·budget_d`, `t_budget = 15·budget_d·T`, `trial_cap = 3·trials_budget`.
  - `capped` mode stops at whichever of (trial_cap, t_budget) comes first.
  - `OPTPLAY_BUDGET_MODE=capped`. Tree tag `<p>@g0.05@bcapped[@timing]`.
- Instances: `optplay/instances/<p>.json` (local), 50 per problem, the thesis uses φ 0-19.
  - Screen (`screen.py:66-152`): 100 θ × 10 reps, a θ passes if all 10 succeed, admit if ≥25 pass. `MAX_ATTEMPTS=2000`.
  - Furuta acceptance ≈2 % (47.6 attempts/φ).
  - Seeds = FNV `substream(index, attempt, …)`, so instances are append-only.
  - φ and noise are stored float64.
- **Bug (confirmed in code):** `run.py:92-96` freezes `phi`, `noise_stds`, `phi_hat` only. Furuta's `kick_voltage`, frozen during screening (`screen.py:117-120`), is **redrawn on every optimization trial**. It is not recorded in `manifest.json`.

### 5.2 Thesis method → cell → code

| Thesis | Cell | Builder |
|---|---|---|
| ESRS | `Random uniform+cost` (`RandomESBO` on Furuta / @timing) | `field.py:110-168`, `RandomUniformOptimizer` + exact running-J > incumbent stop |
| BO | `Plain GP` | `field.py:171-175` → `optimizers.plain_gp.PlainGPOptimizer` |
| ESBO-GP | `Stenger ESBO` | `field.py:178-184` → `optimizers.stenger.StengerBOOptimizer(monte_carlo=True)` |
| ESBO-TabPFN | `ESBO-TabPFN` | **no builder in the local repo** (runs exist on the cluster). Recover from the cluster checkout |
| DecPFN | `MTr 0.9` | `field.py:242-274`, `MatchedTruncatedLivePredictionOptimizer(get_predictor(model), predict_resolution=T//30, matched_threshold=.9, fail_threshold=.9, monte_carlo=False)` |
| DecPFN-H | `HMTr 0.9` | `field.py:213-239`, `HybridMatchedTruncatedLivePredictionOptimizer(monte_carlo="gp", fail_threshold=.9)`, native DecPFN cadence |
| XDecPFN-PI | `QCn PI abs 0.99` / `QC PI abs 0.99` | `field.py:498-546` → `q_loop.PureModelLoopOptimizer(QPredictor(spec), threshold=.99, acq="PI", stop_rule="abs")` |
| FT-PFN-EI | `QF EI abs xgrid 0.99` | `field.py:549-588` → `fr2_q.FR2QPredictor`, `stop_grid="c1r2"` |
| ESBO-Oracle | `Oracle BO` | `field.py:187-191` → `optimizers.oracle_bo.OracleBOOptimizer` |

- Variant figures:
  - `MTr/HMTr {0.9, 0.99, 0.99 ft0.99}`.
  - The qx sweep `{QCn,QF} {EI,PI} [abs] [xgrid] {0.9,0.99,0.999}` (`field.py:669-808`). The violins use only the abs 0.9/0.99 cells.
- Dead cells still in `CELLS`: `HMTr 0.95/0.999`, all `* MC`, native-cadence `QF *`, `M 0.999 noCost`. Caution: they still enter `J_best` (§5.4).

### 5.3 Pure model loop (`q_loop.py`)
- Init: `n_init = max(d,2)` uniform. Uniform draws continue until one trial completes (:457-461).
- Proposals: 2000 candidates = 400 uniform + 1600 Gaussian around the 5 best completed θ, sd 0.10·span, clipped (:170-176, 437-450).
- Scoring: `query_c0` batched, then the argmax of `q_acq.score` (EI/PI on bins × (1−p_crash), incumbent = best completed).
  - Axis `ln_J` for QC/QCn (`q_acq.py:72-89`), `raw_J` for QF (:126-149).
- Context: all past usable trials (cut trials truncated, not imputed), cap `n_max` (unreachable).
- Stop (`early_stop`, :549-628):
  - (a) exact running J > incumbent.
  - (b) `abs` ("naive"): `p_fail + (1−p_fail)·P(J̃>J_inc) > τ`, refreshed when the token index moves.
  - Both are disarmed during the initial design.
  - QF xgrid refreshes on XDecPFN's grid.
- Dead knobs to drop: `bstop`, `mtr` stop for qx, `model_stop`, `logEI`, `HZ`, `seed_offset`, diag env vars.

### 5.4 Runs, provenance, crashes, storage
- `python -m optplay.run --problem P --instances 0-19 [--cell NAME…|--cells-file F] [--workers N]` (`run.py:479-521`).
  - Seed `substream(*problem.encode(), index)`, repetitions 1.
  - Finished `runs/<cell>.json` are reused.
  - Guards: per-cell checkpoint provenance in the manifest (`run.py:349-383`), stop-rule fix key (`:202-246`, prediction commit 76e367e).
- Runs per cell:
  - 20 φ for all cells.
  - abs 0.99/0.999 qx cells have 34-50 φ: cartpole/threetank/pt2 50, boiler 34-35/49, furuta 42-43/49. The crash table therefore uses 43-50 runs for those cells.
  - QCn exists only on boiler/threetank/pt2.
- Raw: `$HPCWORK/optplay/<p>@g0.05@bcapped/phi_NNNN/{runs/<cell>.json, manifest.json}` (~30 GB) and `…@timing/` (3 φ, workers=1, cells from `experiments/timing/cells*.txt`).
- Aggregate: `python -m optplay.aggregate [--problem P] [--crashes]` writes `optplay/results/<tag>_{curves.json,runs.json,runs.csv,final.csv,crashes.json}`.
  - Crash = `failed[t-1]`. `mean_crash_frac` on a 200-point time grid.
  - `J_best` = `curves.json["reference"]`: min successful final cost over **every cell on disk**, beyond 15d trials too (`aggregate.py:196-197`).
- Local `optplay/results/` (Sep 20/22):
  - Stale against the thesis build: run counts differ (cluster_answers fingerprints) and no `ESBO-TabPFN` rows are present, so `reference` may differ.
  - `_crashes.json` (Sep 20) feeds `optplay_crashes`.
  - Legacy `boiler*`, `cartpole_*`, `*@g0.05_*`, `pred_gate_*`: drop.
- Slurm `optplay/slurm/`: `run_problem.slurm` (default BUDGET_MODE=full! pass capped), `run_qx.slurm` (array per φ, sets `C1R2_PROBLEM`), `post_problem.slurm`, `crashes_reaggregate.slurm`, ~40 `*cells*.txt` wave lists. Keep `qcn_sweep_cells.txt`, `qx_abs09_*`, `qx_xgrid*`, `mtr_hmtr_cells.txt`, `escells.txt`.

### 5.5 Figure axes (`thesis_figures/optplay_cross.py`)
- Trials axis: 1..15d (not the 3·15d cap).
- Time axis: `linspace(0, 15dT, 300)`. The incumbent counts the last *finished* trial.
- Per problem: median over the 20 common φ of `(J−J_best)/(J_ESRS−J_best)`, then mean over the 5 problems, then log10. Display floor 1e-4.
- `J_ESRS` = ESRS's end-of-run incumbent.
- Cache `$HPCWORK/optplay/_thesis_cache` (fingerprint keyed). Built by `thesis_figures/slurm_build_optplay_cross.slurm` (5 CPU, 48 G, ~22 min, needs TeX Live). It also renders sheet, violins and variants.

### 5.6 Imports from `impl/prediction`
- `experiments.resolve_dataset`, `experiments.{boiler,threetank_pi}.system`.
- `benchmark.OptimizerBenchmark`, `util.matlab_gp`, `util.mpl_styles`.
- `model.{system.Trial, optimizer.Optimizer, predictor.{Predictor,HistogramPrediction}}`.
- `optimizers.bo_gp.{esbo_incumbent,esbo_n_init,is_completed,trial_final_cost}`, `optimizers.{random_uniform,plain_gp,stenger,oracle_bo}`.
- `optimizers.live_prediction.{MatchedTruncated…,HybridMatchedTruncated…,condition_above,p_value_above…}`.
- `predictors.decpfn.infer.get_predictor`.
- The missing ESBO-TabPFN presumably uses `optimizers.tabpfn_bo`.

### 5.7 Keep/drop
- **KEEP (trim):** `env, problems, screen, run (fix kick), aggregate, q_loop, q_acq, q_predictor, fr2_predictor, fr2_q, field` (≈14 thesis cells + ESBO-TabPFN), `c1_predictor` (c1r2 path only), `model_specs` (C1R2 tables only).
- **DROP:** `field_archive, f_predictor, c1_gate, pred_gate, c1_figure, figures, compare, stop_rule_compare`, `archive/`, `figures/`, `instances/furuta_lqr.json`, `FINDINGS.md` (fold §3.16 ESRS proof into the docs).

### 5.8 Discrepancies (optimization)

| # | Thesis | Code |
|---|---|---|
| O1 | setup.tex:632-634: optimize the screened instance | furuta `kick_voltage` unfrozen in `run.py:92-96` |
| O2 | method.tex:1149-1161: EI/PI uniform in J within a bin | XDecPFN integrates uniform in ln J (`q_acq.py:72-89`) |
| O3 | method.tex:908-910: DecPFN samples the virtual observation | `monte_carlo=False` = moment-matched (`field.py:242-273`) |
| O4 | setup.tex:99-103: all stop on XDecPFN/DecPFN token positions | MTr refreshes every T//30 steps (`field.py:264`) |
| O5 | DecPFN named by τ only | extra crash rule `P(fail)>τ_crash` (`live_prediction.py:426-505`). "τ=0.99" cells are `ft0.99` |
| O6 | setup.tex:638: J_best = best cost any optimizer reached | over ~60 cells incl. non-thesis ones and trials past 15d |
| O7 | setup.tex:632: 20 instances | crash shares use 43-50 runs for the abs qx cells |
| O8 | setup.tex:671-673: times averaged over instances | separate @timing tree, φ 0-2, workers=1 |
| O9 | ESBO-T-TabPFN in tables | builder missing locally |
| O10 | setup.tex:58: N_init random trials | QC/QF keep drawing uniformly until a trial completes |
| O11 | instances/furuta_lqr_rigreset.json `budget_d: 5` | stale metadata. Code uses 4 |

---

## 6. HARDWARE figure scripts (`thesis_figures/hwplay_*`, `hwpred_*`)

- Data:
  - `impl/prediction/hwplay/runs_holdcost5_newbox_reset_fixed/lqr_hardware/` ("without weight", 6 cells × 5 seeds + `qc_pi_0.99_n10k` seeds 2-3 [dropped], 121 MB).
  - `…_reset_fixed_different_plant/lqr_hardware/` ("with weight", 114 MB).
  - Each holds `manifest.json` + `<cell>/seed*/{run.json,trials.jsonl}`. **LOCAL, git-tracked.**
- Code: imports `hwplay.{aggregate,config,figures}` from `impl/prediction/hwplay` (numpy only).
  - Always re-aggregates from `trials.jsonl`. The committed `hwplay/results/*` are stale and not read (`hwplay_optimization.py:61-78`).
  - Cell map `NAME_MAP` (`hwplay_optimization.py:157-187`).

| Script | Output PDFs |
|---|---|
| `hwplay_optimization.py` | `hwplay_optimization__{with,without}_weight` (median log regret vs wall-clock/trials + mean crashes, band 25/75), `hwplay_regret_violin` (:561), `hwplay_timing` (:650), `hwplay_trials_over_time` (:699). Time is reset-normalised by the DecPFN-H mean reset |
| `hwplay_best_controller.py` | `hwplay_best_controller` (combined, WALL_S=250, :50) |
| `hwplay_cost_early.py` | `hwplay_cost_early` (first 300 s, raw cost) |
| `hwplay_earlystop_dial.py` | `hwplay_earlystop_dial__{with,without}_weight` |
| `hwpred_benchmark.py` | `hwpred_benchmark` (+csv). Plot only, reads `experiments/hwpred/results/hwpred_ladder_{reset_fixed,different_plant}.{parquet,json}`, `hwpred_timing_*.json` |
| `hwpred_ablation{,_bars,_curves}.py` | `hwpred_ablation*`, from `hwpred_ablation_*.{parquet,json}` |

- **Missing scripts (cluster-only, uncommitted versions):**
  - `hwplay_acq_wait` (USED, appendix:1065). It is listed in `scripts/build_figures.sh:41` and `style.py:1213`, but the .py file is absent.
  - `hwplay_cost_1000` (USED, hw_results:185) and `hwplay_cost_full`.
  - `hwplay_best_controller__500s{,__with_weight,__without_weight}`: the per-rig ones are USED (hw_results:216, 227).
  - `hwplay_best_controller__{with,without}_weight` (250 s per-rig).
  - `hwplay_optimization__*__minmax`.
  - All read only the local `trials.jsonl` trees. Recover the code from `/rwthfs/rz/cluster/home/dln18500/bachelorarbeit/impl/dynformer/thesis_figures/` or rewrite it.
- hwpred upstream (`experiments/hwpred/hwpred_ladder.py` + `.slurm`, `hwpred_timing.py`) runs inference:
  - Segment-GP via `predictors.stenger.custom`.
  - DecPFN `rigreset_ckpt("deadtime-strat","kick0.6","binned")` through `impl/prediction/furuta_rigreset_models/MANIFEST.json` → `impl/prediction/models/decpfn/furuta/...` (**absent locally**).
  - XDecPFN = furuta seg6 (`field._c1r2_lnj_spec`).
  - Cutoffs 10/30/50/70/90 %.
  - Its results are local, so all hwpred figures re-plot locally.
- `impl/prediction/scripts/compile_hw_trajectories.py` (untracked): dumps trajectories to `hw_trajectories/*.json`. No figure uses it.

---

## 7. LANDSCAPE / PRIOR figures

- `landscape_visualization/plot_landscape_grid.py`: library.
  - Real-rollout landscapes on a random oblique 2-D slice (D>2) or 1-D, blue = cost rank, red = crash probability.
  - Plants from the universal prior in `impl/prediction/experiments/*`.
  - CPU.
- `plot_landscape_context_grid.py --mode {all|worker|render|furuta_sheet|priors_sheet}`:
  - `priors_sheet`: 5 problems × screened instances 0-2 from `optplay/instances`, real landscapes only.
    - `run_priors_sheet.slurm` (3 h, 8 CPU, `--n-slices 3 --grid-2d 24 --grid-1d 80 --repeats 6 --seed 0 --resample cartpole,furuta_lqr_rigreset --slice-seed 1`) writes `out/priors_sheet_cache/priors_sheet.pkl`.
    - Rows were then hand-picked: `slice_candidate.py` (+`run_slice_candidates`, `run_cartpole_slices`, `run_picked_rows` arrays) → `merge_picked_rows.py` (hard-coded `PICKS`). **Not one command.**
  - `furuta_sheet`: one frozen furuta_rigreset plant, 3 slices.
    - Context from two real BO runs (QC PI 0.99 on the seg6 XDecPFN, QF EI 0.99 on `furuta180sg`), model scores at n∈{1,4,16,64}.
    - `run_furuta_sheet.slurm` (2 h, 8 CPU, `--seed 1`). Its output was hand-copied over `out/furuta_sheet_cache/furuta_sheet.pkl`.
    - `run_furuta_samples.slurm` array 0-29 writes `out/furuta_samples/sampleNN/`.
- `thesis_figures/priors_samples.py`: 10 J(t) samples per problem on instance 0. Live CPU rollouts at plot time (local inputs). **Regenerable locally.**
- `thesis_figures/priors_landscapes.py`: re-plot of `priors_sheet.pkl` (+ `furuta_samples/sample02`) — **cluster-only pickle**.
- `thesis_figures/landscape_furuta.py`: 2×5 sheet from `furuta_sheet.pkl` (slice 1). It is not \includegraphics'd.
  - The thesis uses `landscape_furuta_BEST_NO_DELETE.pdf` (sim_results:205). No script writes that name and there is no sidecar, so its **provenance is unknown**. It is likely a `--samples`/`--name` render.
- `priors_landscapes_samples.py`, `priors_slice_candidates.py`, `landscape_visualization/priors_sameslice.py` (untracked, defense slides): not in the thesis.
- `landscape_visualization/out/` is gitignored. Locally only `out/priors_sameslice/` exists.

---

## 8. THESIS FIGURE → SCRIPT → DATA

CLI: `scripts/build_figures.sh` = `python -m thesis_figures.<module>` from the dynformer root (TeX Live pgf, `OPTPLAY_C1R2_CELLS=nmatched`). Outputs go to `thesis_figures/out/`, which is absent locally. They were synced by hand into `writing/thesis/assets_generated/`. Each PDF has a sidecar `.md`.

- **USED** = \includegraphics in `writing/thesis/sections/*.tex`. 31 of 61 PDFs are used.
- **LOC**: L = all inputs on this Mac, C = cluster only, P = partial.
- **REGEN**: can it be re-plotted without recomputing benchmarks or models?

Abbreviations for inputs:
- HW = `impl/prediction/hwplay/runs_holdcost5_newbox_reset_fixed{,_different_plant}/lqr_hardware`
- HP = `experiments/hwpred/results`
- PP = `predplay/results`
- TB = `results_timing_bundle/predplay_timing.json`
- OR = `optplay/results`
- LD = `experiments/ladder/results`
- TC = `$HPCWORK/optplay/_thesis_cache` ← raw `$HPCWORK/optplay/<p>@g0.05@bcapped/phi_*/runs/*.json`
- PR = `$HPCWORK/predplay/preds`
- LV = `landscape_visualization/out`
- PM = `impl/prediction/models`

| PDF (assets_generated/…) | Script | Inputs | LOC | REGEN | USED |
|---|---|---|---|---|---|
| hardware/optimization/hwplay_acq_wait | **missing** | HW | L data | code missing | appendix:1065 |
| …/hwplay_best_controller | hwplay_best_controller.py | HW | L | yes | – |
| …/hwplay_best_controller__500s | **missing** | HW | L data | code missing | – |
| …/hwplay_best_controller__500s__with_weight | **missing** | HW | L data | code missing | hw_results:227 |
| …/hwplay_best_controller__500s__without_weight | **missing** | HW | L data | code missing | hw_results:216 |
| …/hwplay_best_controller__{with,without}_weight | **missing** | HW | L data | code missing | – |
| …/hwplay_cost_1000 | **missing** | HW | L data | code missing | hw_results:185 |
| …/hwplay_cost_early | hwplay_cost_early.py | HW | L | yes | – |
| …/hwplay_cost_full | **missing** | HW | L data | code missing | – |
| …/hwplay_earlystop_dial__{with,without}_weight | hwplay_earlystop_dial.py | HW | L | yes | – |
| …/hwplay_optimization__with_weight | hwplay_optimization.py | HW | L | yes | hw_results:125 |
| …/hwplay_optimization__without_weight | hwplay_optimization.py | HW | L | yes | hw_results:101 |
| …/hwplay_optimization__*__minmax | **missing** | HW | L data | code missing | – |
| …/hwplay_regret_violin | hwplay_optimization.py | HW | L | yes | hw_results:143 |
| …/hwplay_timing, hwplay_trials_over_time | hwplay_optimization.py | HW | L | yes | – |
| hardware/prediction/hwpred_benchmark | hwpred_benchmark.py | HP ladder+timing | L | yes | hw_results:48 |
| …/hwpred_ablation_bars | hwpred_ablation_bars.py | HP ablation | L | yes | hw_results:72 |
| …/hwpred_ablation, hwpred_ablation_curves | hwpred_ablation{,_curves}.py | HP ablation | L | yes | – |
| priors/priors_samples | priors_samples.py | optplay/instances + live CPU sim | L | yes (cheap) | setup:384 |
| priors/priors_landscapes | priors_landscapes.py | LV/priors_sheet_cache/priors_sheet.pkl, LV/furuta_samples/sample02 | C | from pkl | appendix:862 |
| simulation/optimization/at_the_end/optplay_crashes | optplay_crashes.py | OR/<p>@g0.05@bcapped_crashes.json | L (Sep 20, pre ESBO-TabPFN?) | yes | sim_results:476 |
| …/optplay_cross | optplay_cross.py (slurm_build_optplay_cross.slurm) | TC + OR curves.json | C | if TC kept | sim_results:423 |
| …/optplay_cross__sheet | optplay_sheet.py | TC | C | if TC | appendix:967 |
| …/optplay_timing | optplay_timing.py | `$HPCWORK/optplay/<p>@g0.05@bcapped@timing/` | C | raw reread | sim_results:502 |
| …/optplay_violin__cross_full | optplay_violin.py | TC | C | if TC | sim_results:462 |
| …/optplay_violin__sheet | optplay_violin.py | TC | C | if TC | appendix:974 |
| …/optplay_violin__{boiler,cartpole,cross,furuta_lqr_rigreset,pt2_deadtime,threetank_pi} | optplay_violin.py | TC | C | if TC | – |
| simulation/optimization/decpfn_100_binned/optplay_decpfn_thresholds | optplay_variants.py | TC | C | if TC | – |
| …/optplay_decpfn_thresholds__violin | optplay_variants_violin.py | TC | C | if TC | appendix:953 |
| simulation/optimization/ftpfn_xdecpfn/optplay_qx_variants | optplay_variants.py | TC | C | if TC | – |
| …/optplay_qx_variants__violin | optplay_variants_violin.py | TC | C | if TC | appendix:960 |
| simulation/prediction/at_the_end/fig2_sheet | fig2_sheet.py | PP/fig2_{pt2,cartpole,furuta_rr}.csv + PR (noSt curve) | P | cluster | sim_results:308 |
| …/fig3_cross | fig3_per_problem.py | PP fig1_* + TB | L | yes* | sim_results:253 |
| …/fig3_{boiler,cartpole,furuta_lqr_rigreset,pt2_deadtime,threetank_pi} | fig3_per_problem.py | PP + TB (+PR for furuta med_D) | L/P | yes* | – |
| …/fig3_sheet | fig3_sheet.py | PP + TB | L | yes* | appendix:930 |
| simulation/prediction/ftpfn_xdecpfn/context_ladder | context_ladder.py | LD (cartpole/furuta `celllnjw128`, pt2 `cellnmatched`, fr2) | L | yes | sim_results:158 |
| …/ftpfn_xdecpfn_example | prediction_example.py | **live inference**: c1r2 cartpole seg2 + fr2 ckpt + predplay field | C | no | sim_results:198 |
| …/ftpfn_xdecpfn_panel | ftpfn_xdecpfn_panel.py | PP + TB | L | yes* | – |
| …/ftpfn_xdecpfn_stop_sheet | fig2_sheet.py | PP fig2_* | L | yes | sim_results:224 |
| …/landscape_furuta | landscape_furuta.py | LV/furuta_sheet_cache/furuta_sheet.pkl | C | from pkl | – |
| …/landscape_furuta_BEST_NO_DELETE | unknown (hand-renamed) | furuta pkl (unknown) | C | unknown | sim_results:205 |
| simulation/prediction/lcpfn_decpfn/gridsearch_panels | gridsearch_panels.py | PM `models/{decpfn,lcpfn}/cartpole/universal/grid/*/trials.csv` | C | cluster | appendix:522 |
| …/lcpfn_decpfn_example | prediction_example.py | **live inference** (PM ckpts) | C | no | sim_results:75 |
| …/lcpfn_decpfn_heads | lcpfn_decpfn_heads.py | PM `…/transfer/{10,100}/<head>/trials.csv` | C | cluster | – |
| …/lcpfn_decpfn_heads_benchmark | lcpfn_decpfn_heads_benchmark.py | PR + PP | C | cluster | sim_results:82 |
| …/lcpfn_decpfn_stop_sheet | lcpfn_decpfn_stop_sheet.py | PR | C | cluster | sim_results:126 |
| simulation/prediction/tabpfn_stenger_tabpfn/tabpfn_panel | tabpfn_panel.py | PP + TB | L | yes* | sim_results:44 |

`yes*` means the local `fig1_*` is the 2026-09-29 lcpfn_zscore build. Exact thesis numbers need the `*_pre_lcz` files or the 5P CSV (§4.4). The figure scripts read the canonical names.

Other inputs:
- `thesis_figures/sidecar_results/*.md` (39) are hand-written blurbs spliced into sidecars (`style.py:1330`). They are inputs.
- `writing/thesis/data/sources/` holds `cluster_answers_2026-09-24.md`, `fig1_cross_5prob_canonmed.csv` and `hwpred_benchmark.csv`, the local copies of cluster numbers.

---

## 9. SLURM / HPC

- `scripts/slurm_submit.sh`: an sbatch wrapper that prices the job from CLI + `#SBATCH` and refuses >5 GPU-h or >100 core-h per job.
  - Per-run ledgers live in `logs/slurm_budget/`.
  - Env knobs `DYNFORMER_SLURM_MAX_{GPU,CPU}_H`.
  - Agent-workflow tooling: a public repo should call `sbatch` directly or ship plain job scripts.
- Modules (ENVIRONMENT.md §1, §8):
  - Python: `module purge && module load GCCcore/13.3.0 Python/3.12.3`. Without it, numpy segfaults.
  - C++ generators: `module load GCC/13.3.0 imkl/2024.2.0`.
  - No CUDA module: the torch cu128 wheel ships its runtime.
- uv: always `uv run --no-sync --project ../prediction …`. dynformer has no own env.
  - Batch jobs need `export PATH="$HOME/.local/bin:$PATH"`.
  - Concurrent `uv sync` corrupts `.venv`, so use `flock`.
- Accounts:
  - `rwth2055` for CPU (`-p c23ms/c23mm/c23ml`).
  - `thes2287` (`$PROJECT`) for GPU (`-p c23g`, H100, 24 CPU/GPU).
- Thesis model training did **not** use Slurm. It ran on the login node `login23-g-1` (4× H100, GPU 2 MIG-partitioned, select by UUID, 30-min process kill), driven by `c1r2_lane.py` segments / `fr2_run_gpu0.sh` relaunches over ssh.
- Hardcoded paths to parameterise:
  - `/hpcwork/dln18500/...` (`field.py` C1R2 tables, `c1r2_env.py:123`, driver scripts).
  - `/work/dln18500/...` (furuta c1r2 + fr2, `field.py`, `registry.py:601-603`).
  - `/rwthfs/rz/cluster/home/dln18500/bachelorarbeit/impl/dynformer` (driver `ROOT=`, `predplay/slurm/fr2_predict.slurm`).
  - GPU UUIDs (`c1r2_env.py:174-179`, drivers).
  - `$HPCWORK/{c1r2,fr2_F,optplay,predplay,oraclebench}`.
- Env vars a rebuild must eliminate or document:
  - `C1R2_PROBLEM` (import-time binding), `C1R2_VFEATS`, `OPTPLAY_C1R2_CELLS`, `OPTPLAY_C1R2_{CKPT,CALIB,EDGES_SHA256}_<P>`, `OPTPLAY_FR2_CELLS`, `FR2_WORK`, `FR2_PREDPLAY_CKPT_DIR_<P>`.
  - `OPTPLAY_BUDGET_MODE`, `OPTPLAY_TIMING`, `OPTPLAY_CHECKPOINTS`, `OPTPLAY_Q_CHUNK`, `OPTIM_DISABLE_USETEX`.
- Compute recorded:
  - XDecPFN: ~0.5 GPU-h per problem cell.
  - FT-PFN: ~41.5 min cartpole (setup.tex Table), up to several h for furuta incl. restarts.
  - Data generation: c1r2 cartpole tranche ~190 GB, furuta φ screen ~233 core-h, fr2 furuta ~1,790 core-h, fr2 others 12-38 core-h per M contexts.
  - optplay raw tree ~30 GB.

## 10. TESTS and DEPENDENCIES

- **No `pyproject.toml`/`requirements.txt` in dynformer.** It runs in `impl/prediction`'s uv env: `impl/prediction/pyproject.toml` + `uv.lock`.
  - Python ≥3.12,<3.14. torch ≥2.4 (cu128 index on Linux).
  - numpy ≥2.1.3, scipy, pandas, scikit-learn, tabpfn ≥8.0.5, gpytorch, botorch, ax-platform (used by `c1r2_bo.py`).
  - gymnasium, mujoco (furuta experiments import them).
  - `gym_brt` editable from `../furuta/furuta/quanser-driver` (hardware only).
  - matplotlib (via prediction).
  - `flex_attention` needs torch ≥2.5 in practice (FR2).
- C++ generators must be built in `impl/prediction/tools/{cartpole_cpp,plants_cpp,furuta_cpp}`:
  - `cartpole_gen_s11`, `plant_gen_ctx2`, `furuta_gen_ctx2`, `furuta_gen_phi` (GCC 13.3 + imkl).
  - Build instructions are in prediction's `tools/*/gen.slurm`.
- A minimal requirements list for inference only: `~/Downloads/dynformer_timing_bundle/requirements.txt` (torch, numpy, scipy, pandas, scikit-learn, tabpfn, gymnasium, mujoco).
- Tests (plain-python scripts with asserts, not a pytest suite):
  - `experiments/c1r2/*_tests.py` (11 files: leak suite, data, grid, ladder, train, transfer).
  - `experiments/f2/f2_{general_law,grid_rule}_tests.py`, `experiments/s11/*tests*`, `experiments/s12/*tests*`, `experiments/recipe/driver/test_*.py`, `experiments/landscape_sampler/test_*.py`.
  - `predplay/predplay/{c1r2_group_tests,recipe_group_tests,c1r2_prefix_tests}.py`.
  - `experiments/fr2/fr2_check.py` (45 data assertions), `fr2_train_check.py` (mask/NLL/resume), `c1r2_verify.py` (leak suite, tranche integrity).
  - Worth porting to pytest: the leak suite (causal-prefix/cross-group/query isolation exact 0), fr2 mask/resume checks, predplay task-table digest verify, the ESRS exactness check (optplay FINDINGS §3.16).

## 11. OPEN QUESTIONS for Tony

1. **`nmatched` training (pt2/threetank/boiler):**
   - Which exact command, starting checkpoint and LR schedule produced `*_nmatched/ckpt_final/f_s20260931_seg{3,2,2}_final.pt`? Was the cosine re-planned for 16,550 steps or truncated?
   - No script or log is in the repo. Can you pull `experiments/c1r2/out/logs/*nmatched*` and the checkpoints' `step`/`lr` metadata from the cluster?
2. **Checkpoint export:**
   - Copy all 10 thesis checkpoints (5 XDecPFN + 5 FT-PFN) with their calibrations and `metrics.json` off the cluster into an archive. Only furuta XDecPFN, the lnjw128 cells and the FT-PFN cells exist in `~/Downloads`.
   - Also the DecPFN/LC-PFN `impl/prediction/models/...` trees: they are absent locally and needed by predplay, optplay and hwpred.
3. **Exact numbers:**
   - Should the clean repo reproduce the 5P CSV / `*_pre_lcz` prediction numbers that the text quotes, or the canonical 2026-09-29 `fig1_*` (lcpfn_zscore) that the figures currently draw?
   - furuta `med_D` −4.5868 in results_facts matches neither file.
4. **Stale optplay aggregates.** Local `optplay/results/*` predate the thesis build and lack ESBO-TabPFN. Do you want `curves.json`, `runs.csv`, `crashes.json` and `_thesis_cache` re-exported from the cluster? `J_best` depends on every cell on disk.
5. **ESBO-TabPFN builder is missing** from the local `field.py`. Where is that code (cluster checkout, uncommitted)?
6. **Missing hwplay figure code** (`hwplay_acq_wait`, `hwplay_cost_1000/full`, `hwplay_best_controller__500s*`, `__minmax`): recover from the cluster, or rewrite?
7. **`landscape_furuta_BEST_NO_DELETE.pdf`**: which cache and arguments produced it?
8. **Furuta `kick_voltage` not frozen during optimization** (`optplay/optplay/run.py:92-96`). Fix in the rebuild (changes results) or reproduce as-is and document?
9. **Boiler grid stride:** the thesis says 11 (N_grid 64), XDecPFN uses k=10. Which stride did DecPFN/LC-PFN boiler use? Correct the thesis table or note the deviation?
10. **Thesis text fixes or code-faithful docs?**
    - Gradient clipping and AdamW (X1/F1). Time and θ embeddings (X2/X3/F2). Readout skip (X7). q_weight 0.5 (X8).
    - ln-J acquisition (X9/O2). Timing GPU RTX 6000 and single call with extrapolated cells (P2). Mean-then-median score (P1). DecPFN moment matching vs sampling (O3), MTr cadence (O4).
11. **Timing:** the thesis timing for XDecPFN used the raw-J S-54 weights (same architecture). Is re-timing on the stated workstation with warmup and repeats acceptable, even though the thesis numbers would move slightly?
12. **Scope of the clean repo:**
    - Include the XDecPFN HPO pipeline (Ax BO, arch grid, lr ladder), or only the final recipe as one trainer?
    - Include the data generation (needs the C++ generators and ~2,000 core-h for Furuta), or ship the trained checkpoints and frozen benchmark fields?
13. **Drop `furuta_lqr` (plain) everywhere?** It is scored in predplay and influences nothing in the thesis except the 5P med_D mixing.
