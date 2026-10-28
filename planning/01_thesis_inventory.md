# 01 Thesis inventory: everything the code must reproduce

Source of truth: `writing/thesis/sections/*.tex` as included by `main.tex` (introduction, background, related_work, method, setup, sim_results, hw_results, conclusion, appendix). Line numbers are those of the .tex files (approximate, `cat -n`). Abbreviations: `M` = method.tex, `S` = setup.tex, `SR` = sim_results.tex, `HR` = hw_results.tex, `A` = appendix.tex, `B` = background.tex, `I` = introduction.tex. `RF` = data/results_facts.md, `CL` = data/sources/cluster_answers_2026-09-24.md, `SC:<name>` = figure sidecar under assets_generated (Data/Metric sections only), `PR` = thesis/PROBLEMS.md (fact-check tracker, 2026-09-30).

IMPORTANT staleness note: RF (2026-09-25) predates the 2026-09-29 "z-score re-aggregation" (LC-PFN retrained with z-scored inputs as `lcpfn_zscore/epochs100/headbinned`, which shifted every `med_D`). The thesis prediction numbers were updated afterwards (PR "Done": "Prediction numbers in sim_results.tex updated to the z-score re-aggregation (FT 1.59, XDec 1.46, DecPFN 0.99, noStates 0.60, LC-PFN 0.35, Segment-GP 0.37/0.55, TabPFN 0.83/0.77, Segment-TabPFN 0.60/0.26 ...)"). Where RF and the thesis disagree, the thesis value is listed and the RF value is given in brackets. RF's crash table (fractions) and HW ablation table (means) are also not what the thesis quotes (thesis uses crash COUNTS and ablation MEDIANS, see Section 7).

---

## 0. Global definitions (I, B, S)

- Cumulative cost `J_t(θ) = Σ_{s=1}^t j_s(θ)`, stage cost `j_t = j(x_t − x_ref_t, u_t) ≥ 0` (I:129-137). Total cost `J(θ) = J_{T_max}(θ)`. Objective = expected total cost `J̄(θ)=E[J(θ)]` (I:139-150).
- Crash: state leaves safety bounds, trial aborted immediately, `J_t = ∞` from then on (I:151-160, eq:ct:crashed).
- Success: not crashed AND problem-specific criterion (only Furuta has one) (M:404-407, S:234-235).
- Incumbent `θ⁻_n`, `J⁻_n` = best, `J⁺_n` = largest total cost among completed non-crashed trials (B:610-612).
- `exceed` rule (Stenger): stop trial as soon as `J_t(θ) ≥ J⁻_n` (B:638-644).
- Segment-GPs: boundaries `T̃_1<…<T̃_ñ=T_max` = sorted unique termination times of NON-crashed trials incl. T_max; crashed trials give no boundaries/observations but count toward the time budget (B:665-683). Data `D^q_n` (B:689-692). Predicted total = observed + sum of segment posteriors, Gaussian with summed means/variances (B:706-719). Virtual obs resampled every BO iteration from that Gaussian truncated at `≥ J_{T_i}` (B:721-725).
- Crashed virtual obs: `J_virt = min{ max{μ_O(θ), J⁻_n} + 3 σ_O(θ), J⁺_n }` with GP conditioned on complete trials `O_n` (B:727-748, eq:esbo:crashed-virtual-objective).
- Algorithm ESBO-GP (B:774-814, alg:ct:esbo): N_init uniform random trials run without early stopping; loop while `n < N_max` and `Σ T_i ≤ T_budget − T_max` (only continue if a full trial still fits): fit each Segment-GP (MLE), sample E_n, fit objective GP on O_n, build F_n, fit GP on `D_virt = O ∪ E ∪ F`, maximize acquisition (max-value entropy search, wang17, B:766-767, A:390), run θ_{n+1} with exceed rule, refresh incumbent. Return best full-length θ.
- Assumption: at least one initial trial completes without crashing (B:768).

## 1. PROBLEMS

### 1.1 Summary table (S:239-265, tab:setup:problems; A:597-692 tab:app:problems; A:697-713 tab:app:grid)

| Problem (code id) | Controller | d_θ | T_max | Can crash | Success crit. | N_stride | N_grid (LC/Dec) |
|---|---|---|---|---|---|---|---|
| Boiler (`boiler`) | bang-bang (TurnOnDelay) | 1 | 701 | no | no | 11 | 64 |
| Three-Tank (`threetank_pi`) | outer-loop PI (fixed inner pump PI) | 2 | 476 | no | no | 7 | 68 |
| PT2 dead-time (`pt2_deadtime`) | parallel PID w/ derivative filter, log10 params | 3 | 501 | yes | no | 8 | 63 |
| Cartpole (`cartpole`) | two parallel PD (cart, pole), bang-bang force | 4 | 1000 | yes | no | 15 | 67 |
| Furuta (`furuta_lqr_rigreset`) | energy swing-up + arm barrier, then LQR within 20° | 8 | 600 | yes | yes | 9 | 67 |

DecPFN grid: `t_ℓ = ℓ·N_stride` for ℓ=1..N_grid−1, `t_{N_grid}=T_max`, `N_grid = ceil(T_max/N_stride)` (M:742-751). LC-PFN grid: `t_ℓ = floor(ℓ T_max / N_grid)`, ℓ=1..N_grid (M:341-347). Choice: Cartpole stride 15 → 67 points, others chosen so N_grid closest to 67 (S:123-134).

Budgets (S:41-58, eq:setup:budgets): `T_budget = 15·T_max·min{d,4}`, `N_max = 3·15·min{d,4}`, applied simultaneously; `N_init = max{d,2}` (Furuta N_init = 8, uncapped, PR). Derived (SC:optplay_cross): boiler 15 trials-axis / T_budget 10515 / N_max 45; threetank 30 / 14280 / 90; pt2 45 / 22545 / 135; cartpole 60 / 60000 / 180; furuta 60 / 36000 / 180. Trial axis of figures ends at `15·min{d,4}` trials (SR:428,435).

### 1.2 Per-problem details (A:616-688 unless noted)

**Boiler.** Port of MathWorks `sldemo_boiler`. Bang-bang burner `ϑ̇ = ±rate/25`, `ϑ(0)=15 °C`, 8-bit ADC thermometer, tracks `ϑ_ref=20 °C`. θ = TurnOnDelay ∈ [5, 500] s (A:829). No safety bound. State seen: `ϑ̃_t` quantized measured temperature (noisy). Stage cost `½·min(|ϑ_ref − ϑ̃_t|, 2)² + u_t + 8·1[u_t > u_{t−1}]`, `u_t∈{0,1}`, switching penalty 0 at t=1. Prior p: heating rate [0.2, 5.0] Scale(5,5); cooling rate [0.02, 0.5] Scale(5,5) (A:737-738). Noise base 0.03 °C → [0.0075, 0.12] (A:785).

**Three-Tank.** Port of RWTH Three-Tank rig (B102→B103→B104), Torricelli throttles, identified pump/valve maps. Outer PI θ=(K_P,K_I), K_P∈[0,3], K_I∈[0,0.1] (A:832-833), on level error, feeding fixed inner pump-flow PI. Setpoint steps to `V_ref = 3e-3 m³` at t=5 s, logging starts there. No safety bound. States: `V_2, V_3` (levels B102, B103), `Ṽ_4` measured level B104 (noisy). Cost `(V_ref − Ṽ_4)²`. Prior: V101 opening [10,85]% Bounded; V106 opening [10,90]% Bounded; B101 volume [0.040,0.130] m³ Bounded (A:741-743). Noise base 1.7e-4 m³ → [4.25e-5, 6.8e-4] (A:788).

**PT2 dead-time.** `G(s)=K ω0²/(s²+2Dω0 s+ω0²)`, 0.1 s input dead-time, parallel PID with derivative filter, θ=(log10 K_P, log10 K_I, log10 K_D) each ∈ [−1,1] (A:836-838), unit step w=1. Crash: `ỹ_t − w > 2` at a logged step. State: `ỹ_t` measured output. Cost ITAE `τ_t |w − ỹ_t|`, `τ_t=(t−1)·0.1 s`. Prior: gain [0.8,20] Scale(5,5); natural frequency [0.1,2.5] rad/s Scale(5,5); damping [0.1,2.5] Scale(5,5) (A:746-748). Noise base 4e-5 → [1e-5, 1.6e-4] (A:791). Sample time implied 0.1 s.

**Cartpole.** Gymnasium cart-pole, explicit Euler 0.02 s, two parallel PD on cart position and pole angle, θ = (K_P,cart ∈[0.3,3.0], K_D,cart ∈[30,300], K_P,pole ∈[10,100], K_D,pole ∈[150,1500]) (A:841-844). Position goal steps between ±2 m. Derivative = per-step error difference. Force bang-bang (max force, sign of summed output). Crash: cart leaves ±3.5 m or pole leaves ±30°. States: x_cart, ẋ_cart, φ_pole, φ̇_pole (all noisy). Cost `10 tanh²(e_t) + ẋ² + φ² + φ̇²`, `e_t = x_ref − x`. Prior: cart mass [0.125,8.0] Scale(8,8); pole mass [0.0125,0.8] Scale(8,8); pole half-length [0.167,1.5] Scale(3,3); max force [3.33,30.0] Scale(3,3) (A:751-754). Noise bases: 0.004 m, 0.02 m/s, 0.002 rad, 0.02 rad/s (A:794-797).

**Furuta** (S:267-300, A:669-688). QUBE-Servo 2 matched. State `(φ_arm, φ_pend, φ̇_arm, φ̇_pend)`, pendulum angle from upright. Energy swing-up with pump gain μ and energy target Ē (fraction of top potential energy), polynomial barrier voltage gain K_P^barrier pulls arm to center; within 20° of upright LQR with `Q=diag(q1..q4)`, `R=r`, designed from `p̂` (only problem using p̂). θ=(μ∈[40,90], Ē∈[0.70,1.20], q1∈[11.5,12.5], q2∈[4.5,5.5], q3∈[0.5,1.5], q4∈[0.5,1.5], r∈[0.5,1.5], K_P^barrier∈[1.5,3.5]) (A:847-854). Crash: arm continuous angle leaves ±90°. Success: no crash and mean |φ_pend| over final two seconds < φ_up = 5°. Cost `j_t = |φ_pend|(1 + 20(1 − min(|φ_pend|, φ_up)/φ_up))` (up to 21× inside band, no velocity/arm terms) (S:290-297). 5 s at 120 Hz → T=600 (S:458-459). Actuation dead-time drawn per instance, ignored by LQR design. Trials start from brake-then-kick reset, kick amplitude drawn per instance. Prior (A:757-767): resistance [2.1,42] Ω Scale(4,5); torque const [0.0105,0.21] Scale(4,5); back-EMF const [0.007,0.315] Tied(1.5) to torque const; arm mass [0.024,0.76] Scale(4,8); arm length [0.028,0.34] Scale(3,4); arm damping [0.275,8.25]e-4 Scale(10,3); pendulum mass [0.006,0.24] Scale(4,10); pendulum length [0.043,0.516] Scale(3,4); pendulum damping [0.0505,1.515]e-4 Scale(10,3); dead-time [8.33,25.0] ms Bounded (†not in p̂); kick amplitude [0,0.6] V, 30 ms pulse, Bounded (†not in p̂). Noise bases (A:800-802): angles 0.002 rad → [0.0005,0.008]; angular velocity 0.07 rad/s → [0.0175,0.28]; motor voltage 0.1 V → [0.025,0.4] (PR: actually actuation noise).

### 1.3 Universal prior recipe (S:302-380, M:386-466)

- Scale params: LogUniform(p0/f_lo, p0·f_hi) per parameter (S:314-323). Bounded: Uniform on physical domain (S:325-328). Tied: `p_dst = p_src · LogUniform(1/ζ, ζ)`, ζ≈1.5 (S:330-335).
- Sensor noise per channel: `σ_i = σ_base_i · S · η_i`, `S ~ LogUniform(1.5/4, 4/1.5)` once per instance, `η_i ~ LogUniform(1/1.5, 1.5)` per channel → σ_i ∈ [σ_base/4, 4σ_base]; iid Gaussian added every timestep to all channels (S:337-350). (PR: single-channel problems draw differently in code.)
- Estimate: `p̃_i ~ N(p_i, (p_i/10)²)`, `p̂_i = max{0, p̃_i}` (S:372-379).
- Degeneracy screen (M:409-414): run 100 random θ, keep the instance iff ≥ 1/4 of them succeed 10 times in a row ("100/25/10 screen", SC:priors_samples, SC:optplay_cross).
- Single-trajectory prior (M:418-437): sample p, p̂, σ; θ ~ U(Θ); trajectory. Reject unsuccessful trajectories w.p. `max(0, 1 − (1−ρ_s̄)/ρ_s̄)` so ≤ ½ are unsuccessful (ρ_s̄ = unsuccessful rate). NO degeneracy screen.
- Trajectory-set prior (M:439-466): screened system; `n ~ U{1..N_max}`; `ρ'_c ~ U(0, ρ_c)`, `ρ'_{s̄|c̄} ~ U(0, ρ_{s̄|c̄})` (ρ_c = crash ratio, ρ_{s̄|c̄} = unsuccessful-given-not-crashed ratio); `n_c ~ Bin(n, ρ'_c)`, `n_{s̄∧c̄} ~ Bin(n−n_c, ρ'_{s̄|c̄})`, `n_s = rest`; draw θ ~ U(Θ) with rejection until exact counts; per example one prob `~U(0,1)`, each trial independently cut with it; cutoff `~U{1..T}`.
- XDecPFN pool variant (M:1122-1132): sample system + pool of N_max full trajectories (counts drawn once with n=N_max), run stage 1 once, reuse pool for N_reuse examples, each redrawing n ≤ N_max, which are cut and where, plus N_unrun unrun random θ*.
- All priors implemented in C++ (A:876-877). Data pre-generated and stored (S:80-84).
- Figure prior samples: 10 θ per problem on screened instance 0, cap 5/10 unsuccessful (SC:priors_samples).

### 1.4 What predictors see / normalization

- DecPFN/XDecPFN token covariates (M:753-784, eq:method:decpfn:covariates): `c_ℓ = [J_{t_ℓ} − J_{t_{ℓ−1}}, ξ_{t_ℓ}, Σ_{t=t_{ℓ−1}+1}^{t_ℓ} ξ̇_t]`, with `t_0=0, J_0=0`; levels ξ at grid value, rates summed over interval at raw resolution (unit-timestep convention); if no measured rates, `ξ=x`, `ξ̇_t = ξ_t − ξ_{t−1}`. Token = linear embedding of `(J_{t_ℓ}, c_ℓ, t_ℓ)`. No-states variant: `(J, ΔJ, t)` only.
- Channels per problem = the `x_t` rows of tab:app:problems (above). (SC:hwpred_ablation: Furuta DecPFN checkpoint has 13 input columns with states, 3 without; see Ambiguities.)
- Normalization (A:329-341): every channel z-scored with per-channel mean/std estimated from prior samples; time t z-scored over its known range {1..T_max} (LC-PFN, FT-PFN) or {1..T_max−1} (DecPFN, XDecPFN). XDecPFN normalizes and bins ln J, FT-PFN raw J (S:187-192, A:554). Gaussian head target z-scored with the same stats, rescaled at inference (A:374-378).

## 2. DATA GENERATION

| Model | Training data (S:205-223 tab:setup:model-cost) | Notes |
|---|---|---|
| LC-PFN 100k / 1M | 1e5 / 1e6 trajectories | single-trajectory prior; Cartpole cost row |
| DecPFN 100k / 1M | 1e5 / 1e6 trajectories | same |
| XDecPFN | 132,400 trajectory sets, each reused at N_reuse=256 → 33,894,400 partially unique examples (S:194-198) | pools of N_max trajectories, no cutoffs |
| FT-PFN | 3,500,000 freshly drawn trajectory sets (S:194) | |

- Cross-trajectory sets on non-Cartpole problems use N_max trajectories with the same step count (S:173-176).
- RF: "E5 = 10 epochs (10⁵ examples), E6 = 100 epochs (10⁶ examples)". SC:gridsearch_panels: HPO on cartpole `universal` prior, batch 128, resolution 15, pool regenerated 2026-09-20, `generate_pool` time-seeded.
- Sampling cost table (A:879-909 tab:app:prior-cost), one HPC CPU core (Intel Xeon Platinum 8468), C++ priors, time for 10,000 samples:

| Problem | Traj. time | Traj. acceptance | System acceptance | FT-PFN sets | XDecPFN sets |
|---|---|---|---|---|---|
| Boiler | 0.27 s | 1.00 | 1.00 | 1.0 min | 5.7 min |
| Three-Tank | 0.59 s | 1.00 | 1.00 | 3.8 min | 9.8 min |
| PT2 dead-time | 0.57 s | 0.88 | 0.88 | 4.9 min | 20.3 min |
| Cartpole | 0.78 s | 0.65 | 0.64 | 21 min | 25 min |
| Furuta | 45 s | 0.0136 | 0.015 | 5.7 h | 6.0 h |

  Text: "on Furuta, sampling 10,000 trajectory sets takes roughly 6 hours on a single CPU core" (S:199-201). Caption: screen dominates on Furuta, only 1.5% pass.
- Sobol: never mentioned in the thesis. Initial design: "randomly selected"/"sampled uniformly from Θ" (M:87-88, B:778, S:57-58).
- Held-out sets: simulated prediction tasks from ESBO-Oracle runs of 65 trials on screened instances (S:525-536); SC: 50 phi per problem, seeds per problem boiler/threetank 4, pt2 7, cartpole 8, furuta 10; queries at trials {1,3,5,9,17,33,65} → n∈{0,2,4,8,16,32,64}; K=50 matched non-crashed queries per (problem,n), plus 50 crashed on crash-capable problems (SR:30-31). Context ladder field: separate ESBO-Oracle runs of N_max trials, 50 runs whose final trial does not crash (SR:171-172) (SC: 181 trials, 50 phi × 2 seeds, query = trial index 180). Optimization: 20 system instances per problem, fixed seeds, screened (S:632-634). HPO validation: NLL at grid cutoffs between 20% and 99%, raw cost units (S:143-146).
- Seeds: not stated numerically anywhere in the thesis.

## 3. PREDICTORS

Common (A:324-357): `nn.Embedding` for constant embeddings, `nn.Linear` for continuous channels; LN γ=1, β=0; GELU FFN; `d_ff = 2 d_model` always; AdamW, cosine LR schedule with linear warmup decaying to 0 at end, grad-norm clip 1, warmup 25% of training (stated as in released LC-PFN, applies to "all four models"); bf16 autocast on CUDA. All training on one NVIDIA H100 with 8 cores of Xeon Platinum 8468 (S:119-121, A:409). Bar heads: N_bin = 1000 (A:363); bin edges for LC-PFN from samples of full trajectory value range, for all others from horizon samples only (A:360-362); `binned` = equal-mass quantiles of prior samples; `linear` = equal-width; bounded bar distribution, out-of-support targets clamped into outermost bins (A:367-371, M:587-588).

Crash-aware heads (M:553-613): bar head outputs N_bin+1 logits, softmax → a_1..a_N, a_∞; `w_∞ = a_∞`, `w_k = a_k/(1−w_∞)`. Loss `−log w_∞` if crash, else `−log(1−w_∞) − log w_k + log(b_k − b_{k−1})`. Gaussian head: 3 outputs, `μ=o1, σ²=exp(o2), w_∞=sigmoid(o3)`, loss `−log w_∞` or `−log(1−w_∞) + ½ log(2πσ²) + (y−μ)²/(2σ²)`. Justified by Corollary (mixed reference measure ν_A, M:516-551, A:251-315).

### 3.1 LC-PFN (M:647-701; model id `lcpfn/epochs100/headbinned`, figures use `lcpfn_zscore/epochs100/headbinned`)
- Architecture of released LC-PFN unchanged (lcpfn_code): encoder, post-LN blocks, output head = two-layer FFN, widened to crash-aware head (M:661-664, A:346-348).
- Tokens: grid points `(t_ℓ, J_{t_ℓ})`, N_grid as DecPFN (S:130-133). One cutoff index κ ~ U{1..N_grid−1} per BATCH; context first κ points; queries all remaining grid points (N_grid−κ targets) (M:690-695).
- Trajectories crashing before t_κ are removed from the batch; optional requeue to front of queue; thesis setting: NO requeue (S:153-155).
- Inputs z-scored (thesis A:332). PR: HPO was done with old input normalization, then retrained z-scored with same hyperparameters.

### 3.2 DecPFN (M:710-799; id `decpfn/epochs100/headbinned`)
- Causal decoder, no query tokens, crash-aware head on each final token rep → prediction of J(θ) at T_max; total cost never tokenized. Pre-LN blocks + final LN, single linear output layer (M:719-732).
- N_grid−1 tokens on exact-stride grid; covariates as Section 1.4.
- Training: full causal stack over positions t_1..t_ℓ (ℓ = largest grid index ≤ T), head at every position; only score cutoffs before crash; batch loss = mean over all valid predictions in batch (M:786-797).
- DecPFN-no-St (`.../states0`): tokens `(J, ΔJ, t)` only, same configuration as corresponding DecPFN (M:783-784, S:152).

### 3.3 LC-PFN/DecPFN HPO (S:135-157, A:417-529)
- On Cartpole only; two training sizes 100k and 1M; objective: validation NLL for horizon prediction at grid cutoffs 20%-99%, raw cost units.
- Fixed start config (A:468): batch N_batch=128, d_model=512 (d_ff=1024), d_k=d_v=16 per head, N_layer=3, `linear` head.
- Stage 1 sequential (A:485-487): LR ∈ {1.0,1.8,3.2,5.6,10.0}×1e-4 (WD 1e-6, dropout 0); then WD ∈ {1e-6,1e-4,1e-2,1e-1}; then dropout ∈ {0,0.1,0.2,0.35,0.5}.
- Stage 2 grid per head: d_model ∈ {256,384,512,1024} × N_layer ∈ {1,2,3,6,12}; wall-clock cap per run 25 s (100k) / 250 s (1M) (A:470-471). Each head uses best cell of its own grid. Figure fig:app:hpo-gridsearch plots training wall-clock vs validation NLL, marker size ∝ params.
- Final configs (A:497-517, dropout 0 everywhere):

| Model | Size | Head | N_layer | d_model | LR | WD |
|---|---|---|---|---|---|---|
| LC-PFN | 100k | binned | 2 | 512 | 5.6e-4 | 1e-4 |
| LC-PFN | 100k | gaussian | 2 | 384 | 5.6e-4 | 1e-4 |
| LC-PFN | 100k | linear | 3 | 384 | 5.6e-4 | 1e-4 |
| LC-PFN | 1M | binned | 12 | 384 | 3.2e-4 | 1e-4 |
| LC-PFN | 1M | gaussian | 12 | 256 | 3.2e-4 | 1e-4 |
| LC-PFN | 1M | linear | 6 | 512 | 3.2e-4 | 1e-4 |
| DecPFN | 100k | binned | 3 | 1024 | 5.6e-4 | 1e-6 |
| DecPFN | 100k | gaussian | 6 | 512 | 5.6e-4 | 1e-6 |
| DecPFN | 100k | linear | 6 | 1024 | 5.6e-4 | 1e-6 |
| DecPFN | 1M | binned | 6 | 384 | 5.6e-4 | 1e-4 |
| DecPFN | 1M | gaussian | 3 | 256 | 5.6e-4 | 1e-4 |
| DecPFN | 1M | linear | 12 | 384 | 5.6e-4 | 1e-4 |

- Training cost on Cartpole, binned head, H100 (S:214-217): LC-PFN 100k 5.8 M params, 2.46e14 FLOPs, 7.6 s; LC-PFN 1M 15.3 M, 6.62e15, 4.1 min (exceeds 250 s cap, PR); DecPFN 100k 26.2 M, 1.10e15, 10.9 s; DecPFN 1M 7.5 M, 3.22e15, 1.4 min. Text: 1M training "still takes at most 250 s" (SR:152).
- SC:gridsearch_panels: both grids record a 2-objective Pareto front (NLL × training_time), "no source in the tree states which point was carried forward" (the thesis table is the only record).

### 3.4 FT-PFN (M:991-1048, A:536-580; id `dynformer_fr2`)
- Context eq:method:ftpfn:context: for each trajectory grid points `((θ_i, t_ℓ), J_{t_ℓ})` for ℓ ≤ κ_i (`t_{κ_i} ≤ T_i`) plus `((θ_i, T_i), J_{T_i})` if T_i < T_max (PR: code adds it for every trial). Crash `∞` embedded as learned constant e_∞. Grid eq:method:grid with N_grid = 8 (S:186) → ≤ 1,620 tokens on Cartpole/Furuta (N_max·(N_grid+1)).
- Pre-LN + final LN, single linear output layer, crash-aware binned head, raw J target.
- Training: queries at horizon for every trial that neither crashed nor completed in context, plus N_unrun = 16 unrun θ (no cutoff). No queries before horizon. Loss: mean within example, then mean over batch.
- Config: LR 1e-4, dropout 0, WD 0, d_model 512, 6 layers, 4 heads, 13 M params, batch 32, 3,500,000 examples, 109,375 steps (A:545-559). Cost Cartpole 1.95e17 FLOPs, 41.5 min (S:219).
- Early stopping queried only at XDecPFN/DecPFN's N_grid−1 token positions (S:97-102).

### 3.5 XDecPFN (M:1057-1132; id `dynformer_c1r2/cellnmatched`; code cell lnJ_W128_qw05, see Ambiguities)
- Stage 1: causal decoder as DecPFN, per trajectory, same grid/N_grid−1 tokens, cacheable. Final token of each trajectory + `e_θ(θ)` + one of `e_complete/e_cut/e_crash`. Unrun query tokens `e_θ(θ*) + e_unrun`. Stage 2: bidirectional self-attention across trajectory tokens; attention mask: no token attends to query tokens. Pre-LN, final LN, single linear crash-aware binned head on trajectory tokens (horizon) and unrun tokens.
- Attention cost O(n² + n N_grid²); incremental O(n² + N_grid²) (M:1095-1099, 1176-1182).
- Training batch: N_sys = 8 systems × N_reuse = 256 examples, N_unrun = 8; mean of all prediction losses across batch (M:1131-1132, A:556-557).
- Config (A:545-559): LR 1e-3, dropout 0, WD 0, d_model 384, 3 causal + 6 cross layers, 6 heads, 11 M params, target ln J, grid as DecPFN, 132,400 examples, 16,550 steps. Cartpole 2.06e17 FLOPs, 32.1 min (S:218).
- FLOPs matched at ≈2e17 on Cartpole (S:169-179); per-problem FLOPs (A:569-579): Boiler X 5.79e16 / F 4.97e16 (0.86), Three-Tank 1.05e17/1.04e17 (0.99), PT2 1.52e17/1.33e17 (0.87), Cartpole 2.06e17/1.95e17 (0.95), Furuta 2.06e17/2.21e17 (1.07). FLOPs measured via `torch.utils.flop_counter.FlopCounterMode` over a few steps.

### 3.6 TabPFN-3 (M:213-361, S:86-90; ids `tabpfn/resolution{L}/typeraw`, `stengertpfn/mode{rate,split}`)
- Released TabPFN-3 regressor (grinsztajn26), default parameters, no fine-tuning, off the shelf.
- Segment-TabPFN `split` (M:256-269): replaces each GP_q by TabPFN with context D^q_n, query θ_i, sample each missing segment independently. `rate` (M:271-305): one shared context of `((θ_k, q), J_rate)` with `J_rate` = segment cost / segment length, all segments, scaled back by length, sampled per segment. Optimizer `ESBO-S-TabPFN`.
- Trajectory-TabPFN (M:325-361): context `((θ_k, t_ℓ), J_{t_ℓ}(θ_k))` for non-crashed trials k, grid points t_ℓ ≤ T_k on eq:method:grid with N_grid = L ∈ {25,50,100,250}; query `(θ_i, T_max)`. Optimizer `ESBO-T-TabPFN` (thesis results name it `ESBO-T-TabPFN`, code cell `ESBO-TabPFN`), selected L = 50.
- Not evaluated at n < 2 for untrained context-aware predictors (SR:23); TabPFN panel draws n≥2 (Trajectory) / n≥4 (Segment).

### 3.7 Segment-GP / ESBO-GP port (A:380-390; id `stenger`)
- Hand-rolled exact GP in PyTorch, constant mean, ARD squared-exponential kernel, hyperparameters in natural-log units (GPML layout) fit with Adam; acquisition MES (wang17). (Background introduces Matérn generally, B:158-192, but the port is SE.) PR "ignored 33": deviates from Stenger (zero mean on normalized outputs, bounded length scales, GPML minimize). GP_COLD_REFIT_GROWTH: thesis silent, sim field uses `gp_cold_refit_growth=0.05` on all five problems (SC:optplay_cross).

### 3.8 Oracle
- `ESBO-Oracle` = ESBO-GP with unchanged stopping rule, virtual obs = point mass at true total cost (S:606-610), so it fits no Segment-GPs (S:536). Excluded from rig (S:625-628).

## 4. STRATEGIES / OPTIMIZERS

Common: initialization and budgets as ESBO-GP (M:84-90). Virtual obs for early-stopped trial: sample from predictor distribution conditioned on `J̃ ≥ J_{T_i}`; finite → virtual obs; ∞ sampled → crash rule eq:esbo:crashed-virtual-objective (M:129-140).

| Thesis name | Definition | Code cell sim / hw |
|---|---|---|
| `ESRS` | θ_n ~ U(Θ), exceed rule vs J⁻_n (S:598-603) | `Random uniform+cost` (`RandomESBO` on Furuta) / `random_stenger` |
| `BO` | same BO (objective GP + MES), no early stopping (S:604-605) | `Plain GP` / `plain_gp` |
| `ESBO-GP` | Stenger, Section 0 | `Stenger ESBO` / `stenger` |
| `ESBO-Oracle` | Section 3.8 | `Oracle BO` / — |
| `ESBO-S-TabPFN` (split/rate) | Segment-TabPFN replaces Segment-GPs, exceed only | not run in optimization |
| `ESBO-T-TabPFN` | Trajectory-TabPFN L=50 replaces Segment-GPs, exceed only (SR:345-348, tab:sim:optimizers) | `ESBO-TabPFN` / — |
| `DecPFN` (`LC-PFN` analog) | exceed + `matched` rule, τ=0.9; virtual obs sampled from DecPFN horizon distribution `p(·|P_{1:T_i})` (M:908-911, 929-935); DecPFN 1M binned | `MTr 0.9` / `mtr0.99_k06binned` |
| `DecPFN-H` | as DecPFN but `hybrid` virtual obs: per early-stopped trial each BO iteration take whichever of Segment-GP (updated) and DecPFN (static, conditioned on J≠∞) has lower variance (M:913-927), τ=0.9 | `HMTr 0.9` / `htmr0.99_k06binned` |
| `XDecPFN-PI` | pure model loop: XDecPFN is only surrogate, no GP; PI acquisition; exceed + `naive` rule with τ=0.99 | `QCn PI abs 0.99` (`QC PI abs 0.99` on cartpole/furuta) / `qc_pi_0.99` |
| `FT-PFN-EI` | pure model loop with FT-PFN, EI, τ=0.99 | `QF EI abs xgrid 0.99` / — |
| Other tested | XDecPFN-EI, FT-PFN-PI, τ ∈ {0.9,0.99} | `QCn EI abs 0.9` etc. |

- `naive` rule (M:855-873): stop when `Pr_{J̃~p(·|P_{1:t})}[J̃ ≥ J⁻_n] > τ` (crash mass counts as ≥). In pure loops predictive is conditioned on H_n, θ_{n+1}, and prefix (M:1167-1172).
- `matched` rule (M:884-906): stop when `Pr[J̃(θ_{n+1}) ≥ J̃(θ⁻_n) | J⁻_n ≤ J̃(θ⁻_n)] > τ`, both predicted at the same cutoff t from their own prefixes, incumbent prediction truncated at J⁻_n; incumbent predictions reused from when it ran. (PR: code keeps the incumbent's crash mass, uses a Gaussian moment instead of a sample, forecasts every ceil(T_max/30) steps, and has a separate τ_crash.)
- Pure loop acquisition (M:1146-1161): bar head; for k: `c_k = max(min(J⁻_n, b_k), b_{k−1})`; `α_EI = (1−w_∞) Σ_k w_k (c_k − b_{k−1})(J⁻_n − (b_{k−1}+c_k)/2)/(b_k − b_{k−1})`; `α_PI = (1−w_∞) Σ_k w_k (c_k − b_{k−1})/(b_k − b_{k−1})`. (PR: XDecPFN bins ln J, code uses log-space EI, PI clamp needs ln J⁻_n.)
- Candidate set (S:104-115): 400 uniform from Θ + 1600 Gaussian: axis-aligned perturbations of one of the 5 best completed controllers, std = 0.10 × extent of Θ per dimension, clipped to Θ. Maximize closed-form acquisition over these 2000.
- FT-PFN stop decisions only at the N_grid−1 XDecPFN token positions (S:97-102).
- Budget caps: N_max = 3·15·min{d,4} trials and T_budget = 15·T_max·min{d,4} steps (code tag `bcapped`); one run per instance, 20 instances per problem (S:632-634). Thresholds tested τ ∈ {0.9, 0.99}; selected τ = 0.9 for DecPFN/DecPFN-H, τ = 0.99 for pure loops (SR:390-413). "ft0.99" cells = τ_crash = τ (RF/CL); the thesis never defines τ_crash.
- Table tab:method:predictor-capabilities (M:1194-1241) lists all possible optimizers incl. `LC-PFN(-H)`, `XDecPFN-EI`, `FT-PFN-PI`. Table tab:sim:optimizers (SR:352-373) lists what is run.

## 5. SIMULATION BENCHMARKS & FIGURES/TABLES

Prediction metrics (S:542-576): conditioned LL on non-crashed queries, conditioned on `J ≥ J_t` (not on non-crash, crash mass kept); five cutoffs 10/30/50/70/90% of T_max snapped to closest grid point before; per-trial score = mean of the five LLs; aggregate = median (per problem; per (problem, n) for context-aware); normalization: subtract per-problem median of every participating score (`med_D`) → `ll_norm`, then median across problems; plotted against median inference time per prediction on workstation GPU; context-aware predictors drawn as Pareto front over context sizes. Crash metric: replay predictor on its own grid, stop when crash probability reaches τ; mean steps saved on crashed trials vs false-stop rate on non-crashed; τ swept over all emitted probabilities. Optimization metrics (S:630-681): simple regret `r = J⁻_n − J_best` (J_best = best cost any optimizer reached on that instance); scaled `r / (J_ESRS − J_best)` per instance (J_ESRS = ESRS's final best on the same instance); median over instances per problem; mean of medians over problems; log10 at the end; x = % of trial budget or % of simulated-time budget. Crashes: mean number crashed so far. Computation time: optimizer decision time summed per run, averaged; sim on one HPC CPU core.

| Label | File (assets_generated/… unless `assets/`) | Content / data | Metric | Type |
|---|---|---|---|---|
| fig:esbo-early-stopping-segments (B:650-663) | assets/esbo_early_stopping_segments | illustration | – | schematic (static) |
| fig:esbo_stenger_extension (M:142-148) | assets/esbo_stenger_extension | illustration | – | schematic |
| fig:method:decpfn-vs-lcpfn-training (M:801-825) | assets/lcpfn_training.pdf, assets/decpfn_training.pdf | illustration | – | schematic |
| fig:method:xdecpfn-training (M:1105-1119) | assets/xdecpfn_training.pdf | illustration | – | schematic |
| tab:method:predictor-capabilities (M:1194) | – | predictor groups/optimizers | – | table |
| tab:setup:model-cost (S:205) | – | params, FLOPs, wall-clock | – | table |
| tab:setup:problems (S:239) | – | problem summary | – | table |
| fig:setup:prior-samples (S:382-394) | priors/priors_samples.pdf | 10 cumulative-cost trajectories per problem, screened instance 0, ≤5/10 unsuccessful | J_t linear, no ticks | 5-panel row |
| fig:setup:Furuta-rig (S:411-432) | assets/qube_pendulum/{no_weight,weight}_angles | photos | – | static |
| tab:setup:prediction-registry (S:495-517) | – | models/axes scored | – | table |
| fig:sim:tabpfn-panel (SR:42-47) | simulation/prediction/tabpfn_stenger_tabpfn/tabpfn_panel.pdf | Trajectory-TabPFN L∈{25,50,100,250}, Segment-TabPFN rate/split, 5 problems, n∈{2..64} | pooled ll_norm vs GPU time, Pareto per variant | cross (Pareto) |
| fig:sim:lcpfn-decpfn-example (SR:73-78) | …/lcpfn_decpfn/lcpfn_decpfn_example.pdf | Cartpole trial run_phi0008_s00 trial 64 (median-cost non-crashed N=64 task, true J=6649.86), cutoffs 35/65/85%, LC-PFN vs DecPFN 1M binned | predictive density, crash % | example (6 flush panels) |
| fig:sim:lcpfn-decpfn-heads (SR:80-85) | …/lcpfn_decpfn_heads_benchmark.pdf | DecPFN, no-St, LC-PFN × {binned,gaussian,linear} × {100k,1M} × 5 problems | median LL non-crashed, pooled over n and cutoffs, centered per problem on mean of drawn arms (caption says median) | dot sheet |
| fig:sim:lcpfn-decpfn-stop-sheet (SR:124-134) | …/lcpfn_decpfn_stop_sheet.pdf | 3 models binned, 100k (faint) / 1M; PT2, Cartpole, Furuta | steps saved vs false-stop ≤0.1 | stop_sheet |
| fig:sim:context-sweep (SR:156-166) | …/ftpfn_xdecpfn/context_ladder.pdf | FT-PFN vs XDecPFN, fixed query (trial 181 of N_max-trial Oracle runs, 50 queries), rungs PT2 {0,1,2,4,8,16,33,67,134}, Cartpole/Furuta {0,1,2,5,11,22,45,90,180} | MEAN conditioned LL (not normalized) | ladder |
| fig:sim:ftpfn-xdecpfn-example (SR:196-201) | …/ftpfn_xdecpfn_example.pdf | same Cartpole trial, cutoff 35%, context n=0,5,10 random trials of same run | density | example |
| fig:sim:landscape-furuta (SR:203-208) | …/landscape_furuta_BEST_NO_DELETE.pdf | one Furuta plant, one 2-D oblique slice (24×24, 6 noisy repeats real), contexts n=1,4,16,64 prefixes of one BO run; XDecPFN top, FT-PFN bottom | mean total cost (rank color) + crash prob | landscape |
| fig:sim:ftpfn-xdecpfn-stop-sheet (SR:222-232) | …/ftpfn_xdecpfn_stop_sheet.pdf | per n∈{0..64}, 50 crashed queries per n | steps saved vs false-stop | stop_sheet |
| fig:sim:fig3-cross (SR:251-256) | …/at_the_end/fig3_cross.pdf | DecPFN, no-St, LC-PFN (points), FT-PFN, XDecPFN, Trajectory-TabPFN L=50, Segment-TabPFN split, Segment-GP (Pareto) | pooled ll_norm vs GPU time | cross |
| fig:sim:fig2-sheet (SR:306-316) | …/at_the_end/fig2_sheet.pdf | XDecPFN, FT-PFN at n=0,64; DecPFN, no-St, LC-PFN pooled | steps saved vs false-stop ≤0.1 | stop_sheet |
| tab:sim:optimizers (SR:352-373) | – | optimizers and tested configs | – | table |
| fig:sim:optplay-cross (SR:421-430) | simulation/optimization/at_the_end/optplay_cross.pdf | 9 methods × 5 problems × 20 instances | log10 of mean-over-problems of per-problem median scaled regret vs % trials (15·min{d,4}) and % sim time | cross (2 panels) |
| fig:sim:optplay-violin-cross-full (SR:460-465) | …/optplay_violin__cross_full.pdf | 9 methods, 100 runs pooled, 50% and 100% sim time | log10 scaled regret, floor −4 | violin |
| fig:sim:optplay-crashes (SR:474-484) | …/optplay_crashes.pdf | 9 methods, PT2/Cartpole/Furuta | mean crashed-trial COUNT vs % sim time | sheet (3 panels) |
| fig:sim:optplay-timing (SR:500-505) | …/optplay_timing.pdf | 7 methods (no ESRS/Oracle), 5 problems, mean of 3 full-budget runs on instances 0-2, 1 CPU core | optimizer time split into GP acq / Transformer acq / Segment-GP forecast / Transformer forecast | stacked bars |
| tab:app:dynamical-foundation-models, tab:app:tsfm-comparison (A:5-200) | – | literature | – | table (no code) |
| tab:app:compute (A:399-414) | – | machines | – | table |
| tab:app:hpo-grid-axes, tab:app:hpo-stages, tab:app:hpo-configs (A:461-518) | – | HPO | – | tables |
| fig:app:hpo-gridsearch (A:520-529) | simulation/prediction/lcpfn_decpfn/gridsearch_panels | all grid cells, 100k / 1M, Cartpole | training wall-clock vs val NLL(20-99), size ∝ params | scatter |
| tab:app:ftpfn-xdecpfn-configs, tab:app:matched-flops (A:536-580) | – | configs, FLOPs | – | tables |
| tab:app:problems, tab:app:grid, tab:app:prior-phi, tab:app:prior-noise, tab:app:theta-box (A:597-858) | – | problems | – | tables |
| fig:app:prior-landscapes (A:860-865) | priors/priors_landscapes.pdf | 5 problems × 3 instances, 2-D slice (Boiler 1-D), grid 24×24, 6 repeats | cost rank (blue) + crash prob (red) | landscape grid |
| tab:app:prior-cost (A:879-909) | – | sampling cost | – | table |
| fig:app:fig3-sheet (A:928-933) | …/at_the_end/fig3_sheet.pdf | fig3 per problem + pooled | ll_norm vs time | sheet (2×3) |
| fig:app:optplay-decpfn-thresholds-violin (A:951-956) | simulation/optimization/decpfn_100_binned/optplay_decpfn_thresholds__violin.pdf | MTr 0.9, MTr 0.99, HMTr 0.9, HMTr 0.99 (τ_crash 0.9), 100 runs, 50/100% | log10 scaled regret | violin |
| fig:app:optplay-qx-variants-violin (A:958-963) | simulation/optimization/ftpfn_xdecpfn/optplay_qx_variants__violin.pdf | 8 pure-loop cells (2 models × EI/PI × τ), naive rule, 100% | log10 scaled regret | violin |
| fig:app:optplay-cross-sheet (A:965-970) | …/optplay_cross__sheet.pdf | per-problem sim-time curves + pooled | log10 median scaled regret | sheet (6 panels, shared y) |
| fig:app:optplay-violin-sheet (A:972-977) | …/optplay_violin__sheet.pdf | per problem, 20 dots | log10 scaled regret | violin sheet |
| alg:app:hw-reset (A:993-1015) | – | reset algorithm | – | algorithm |
| tab:app:hwpred-counts (A:1035-1054) | – | queries per n | – | table |
| fig:app:hw-acq-wait (A:1063-1072) | hardware/optimization/hwplay_acq_wait.pdf | 5 methods (no ESRS), 2 rigs, 5 seeds | mean wait per trial | bars + dots |

## 6. HARDWARE (S:407-484, HR, A:986-1072)

- Rig: Quanser QUBE-Servo 2 rotary inverted pendulum, 5 s trials at 120 Hz (T=600); same controller, Θ, stage cost, crash and success criteria as simulated Furuta; prior defaults are the rig's nominal estimate (except actuation dead-time, unknown) (S:451-461). Same nominal p̂ for both configurations.
- Configurations: `without weight` (tree `runs_holdcost5_newbox_reset_fixed`, code tag `reset_fixed`) and `with weight` = 5 g tip mass at pendulum end (tree `..._different_plant`, tag `different_plant`) (S:463-472, SC). Names not explained in thesis; inferred: `holdcost5` = Furuta hold cost with φ_up = 5°, `newbox` = the Θ box of tab:app:theta-box, `reset_fixed` = the fixed reset procedure of alg:app:hw-reset.
- Workstation: Intel Xeon W-1290P, NVIDIA Quadro RTX 4000 (8 GB) runs rig and optimizers (A:411). Next θ selection starts after reset finishes (no overlap) (S:481-483).
- Reset (A:993-1022): brake pendulum until |φ̇_pend| and angle from hanging small for 0.4 s; coast u=0 for 1.5 s (reopen board if frozen); repeat PD to center arm (K_P=3.0 V/rad, K_D=0.2 V s/rad, at least |u|=0.35 V while stuck) until |φ_arm|<2° or 3 s or moving away; coast 0.3 s; repeat {recenter if |φ_arm|>5°; brake, hold 0 V} until for 2 s continuously ||φ_pend|−π|<2°, |φ̇_pend|<0.05, |φ̇_arm|<0.05 rad/s; zero φ_arm; draw s∈{−1,+1} from trial seed; kick u = s·0.6 V for 0.03 s; start trial. Connection reset every tenth trial. Reset-normalized time: every trial's reset time replaced by the mean reset time of the DecPFN-H runs on that setup (9.28 s without, 9.33 s with weight; SC/RF).
- Optimizers on rig (HR:17-27): ESRS, BO, ESBO-GP, DecPFN, DecPFN-H, XDecPFN-PI; 5 seeds per setup (S:666), seeds shared across techniques (same initial design, SC). Budget: N_max = 180 trials, T_budget = 36000 steps (min{d,4}=4; SC: manifest records 180 trials / 36000 steps). Regret unscaled log10, J_best = best cost recorded on that setup (SC: 256.541 without, 256.917 with weight). Wall-clock axis = reset-normalized time; trial axis first 60 trials (HR:107).
- Hardware checkpoints (SC:hwpred_benchmark): DecPFN = `deadtime-strat / kick0.6 / binned` (13 input columns, with states; this is "k06binned"); XDecPFN = c1r2 ln-J cell (weights of `qc_pi_0.99`). `qc_pi_0.99_n10k` cell exists on disk, dropped.
- Hardware prediction benchmark (HR:29-68): queries = trials not stopped early (crashed or not), not first of run; context = n immediately preceding trials whatever outcome; n ∈ {0,1,2,5,11,22,45,90}, scored only if ≥ n predecessors; predictors Segment-GPs, DecPFN, XDecPFN; LL only. Figure fig:hw:pred-benchmark (`hardware/prediction/hwpred_benchmark.pdf`): median conditioned LL non-crashed (not normalized) vs inference time on workstation GPU (SC says measured on CPU), Pareto fronts, medians below −7 omitted. Query counts tab:app:hwpred-counts: without 757,757,725,632,507,422,268,58; with 694,694,664,574,454,346,193,50. XDecPFN drops queries ending before first grid point.
- Ablation fig:hw:pred-ablation-bars (`hwpred_ablation_bars.pdf`, HR:70-95): 4 DecPFN 1M binned models {dead-time in prior, no dead-time} × {states, no states}; full history context (cap 180); median LL non-crashed (479 / 508 queries); y-axis not from 0. (PR: ablation uses `deadtime-unif`, not the deployed `deadtime-strat`.)
- Optimization figures: fig:hw:optimization-without-weight / -with-weight (`hwplay_optimization__{without,with}_weight.pdf`): (a) median log10 regret vs wall-clock, (b) vs first 60 trials, 25/75% seed bands, (c) mean crashed count vs wall-clock. fig:hw:regret-violin (`hwplay_regret_violin.pdf`): per-seed log10 regret at 50/100% wall-clock budget, seed defining J_best excluded. fig:hw:cost-early (`hwplay_cost_1000.pdf`): median best successful raw cost over first 1000 s, 25/75% band. fig:hw:best-controller-{without,with}-weight (`hwplay_best_controller__500s__*.pdf`): incumbent at 500 s of median seed, arm angle and unwrapped pendulum angle over one rig execution. fig:hw:swingup (`assets/qube_pendulum/swingup.png`). fig:app:hw-acq-wait.
- Success predicate in hwplay figures: SC says `successful` = no crash and final sample within 20° (not the thesis's 5°-mean-over-2-s `held` test). See Ambiguities.

## 7. NUMBERS TO REPRODUCE

### 7.1 Off-the-shelf (SR:54-69; source 5P/TB, thesis values are post z-score)
| Claim | Thesis | RF (stale) |
|---|---|---|
| Segment-TabPFN rate vs split at n=64 | 0.26 vs 0.60 nats | 0.325 vs 0.638 |
| rate vs split time n=64 | 0.37 s vs 4.20 s | 0.367 / 4.197 |
| Trajectory-TabPFN L=100 peak / L=50 | 0.83 / 0.77 nats | 0.909 / 0.816 |
| L=100 time ≈ 2× L=50 | (1.29 s vs 0.61 s, RF) | |
| L=250 no better than L=100 | qualitative | 0.746 |

### 7.2 Single-trajectory (SR:95-152)
- Strict ranking DecPFN > no-St > LC-PFN in every (problem, size, head) except gaussian 100k PT2 (no-St beats DecPFN). LC-PFN gaussian on PT2 does not improve with size.
- 1M binned: DecPFN over LC-PFN 0.45 to 0.79 nats; no-St over LC-PFN 0.08 to 0.33; DecPFN over no-St 0.19 to 0.70 except PT2 0.14 (Boiler 0.19). [RF stale: 0.47-1.20, 0.11-1.05.] Raw DecPFN/no-St values per problem (unchanged by z-score): Boiler −3.6415/−3.8311, Three-Tank 10.7244/10.0195, PT2 −3.3271/−3.4712, Cartpole −6.8184/−7.1580, Furuta −3.7392/−4.3285 (SC:heads_benchmark).
- DecPFN 1M binned best head everywhere except Furuta (linear better by 0.003: −3.7361 vs −3.7392).
- Stop sheet: ordering holds for 1M everywhere and 100k on Cartpole; on PT2 and Furuta 100k models indistinguishable, on Furuta with LC-PFN 1M (≈30 steps at false-stop 0.1). [RF: DecPFN 1M 63/155/71, no-St 58/≈100/35, LC 51/60/25 steps — LC values stale.]
- 1M training ≤ 250 s.

### 7.3 Cross-trajectory (SR:182-243; SC:context_ladder per-rung)
| n | PT2 XDec | PT2 FT | Cart XDec | Cart FT | Furuta XDec | Furuta FT |
|---|---|---|---|---|---|---|
| 0 | −3.370 | −4.099 | −7.289 | −7.247 | −4.294 | −4.203 |
| 16/22 | −2.236 | −2.720 | −6.203 | −6.767 | −2.923 | −2.929 |
| 33/45 | −2.185 | −2.860 | −6.215 | −6.792 | −2.894 | −2.842 |
| top 134/180 | −2.255 | −3.142 | −6.191 | −6.779 | −2.843 | −3.049 |
Thesis: FT worsens after n=16 (PT2) and n=45 (Furuta); XDec improves every step on Furuta, ends ahead; ends ≈0.6 (Cartpole) and ≈0.9 (PT2) nats ahead; n=0 Cartpole FT −7.25 vs XDec −7.29. Stop sheet (RF/SC, at false-stop ≈0.05-0.06): XDec n=0→64 Furuta 72.7→178.9, PT2 82.8→70.5, Cartpole 210.8→366.6; FT 16.5→91.0, 55.1→41.1, 202.2→277.1.

### 7.4 Key representatives (SR:264-334)
| Claim | Thesis | RF (stale) |
|---|---|---|
| FT-PFN n=64 pooled | 1.59 | 1.618 |
| XDecPFN n=64 pooled | 1.46 | 1.524 |
| Per problem XDec vs FT: PT2 | 3.21 vs 2.97 | 3.473 vs 3.233 |
| Cartpole | 1.61 vs 1.46 | 1.617 vs 1.462 |
| Furuta | 2.23 vs 2.09 | 2.235 vs 2.092 |
| FT wins Boiler, Three-Tank | qualitative | 1.458/1.112, 1.467/1.139 |
| DecPFN (context-blind) | 0.99 | 1.038 |
| FT & XDec from n=2 at least | 1.09 | 1.14 |
| best untrained (TabPFN L=50 n=64) | 0.77 | 0.816 |
| DecPFN-no-St | 0.60 | 0.734 |
| LC-PFN | 0.35 | 0.218 |
| n=0 FT / XDec | 0.17 / 0.61 | 0.216 / 0.643 |
| Segment-GP beats LC-PFN first at n=16 | 0.37 | 0.413 |
| Segment-GP n=64 | 0.55 (< no-St) | 0.594 |
| TabPFN beats Segment-GP at n=64 by | 0.22 | 0.22 |
| TabPFN beats LC-PFN from n=16 | 0.58 | from n=8 (0.324) |
| Segment-GP at n=2 | −15.3 | −15.349 |
| Times Segment-GP / TabPFN L=50 at n=16, 32, 64 | 0.21/0.38, 0.77/0.38, 1.32/0.61 s | same |
| Segment-TabPFN split n=16..64 | 0.68 to 4.20 s | same |
fig2_sheet (SR:322-334): XDec PT2 82.8 (n=0) vs 70.5 (n=64); Furuta XDec up to 204.2 vs FT n=64 120.7 (false-stop ≤0.1); Furuta single-trajectory 65.4 (DecPFN), 21.7 (no-St), 19.8 (LC-PFN, RF stale 15.5) vs FT n=0 19.3. Full RF table: XDec n=0 82.8/274.7/76.8, XDec n=64 70.5/396.3/204.2, FT n=0 60.1/226.6/19.3, FT n=64 41.1/314.8/120.7, DecPFN 62.9/146.9/65.4, no-St 56.0/98.2/21.7, LC 50.8/50.8/15.5 (PT2/Cartpole/Furuta; LC values stale).

### 7.5 Optimization (SR:390-553; CL)
- DecPFN thresholds (violin medians, 100 runs, 50% / 100%): MTr 0.9 −0.074/−0.253 (6 at floor); MTr 0.99 (τ_crash 0.9) −0.098/−0.276; MTr 0.99 ft0.99 −0.089/−0.276; HMTr 0.9 −0.087/−0.347; HMTr 0.99 (τ_crash 0.9) −0.057/−0.325; HMTr 0.99 ft0.99 −0.048/−0.286. Thesis: overlap, chose τ=0.9.
- Pure loops at 100% (naive): XDec EI 0.9 −0.656, EI 0.99 −0.726, PI 0.9 −0.736, PI 0.99 −0.824; FT EI 0.9 −0.585, EI 0.99 −0.758, PI 0.9 −0.526, PI 0.99 −0.585. Thesis: all in [−0.82, −0.53], 0.99 beats 0.9 everywhere, PI for XDec, EI for FT.
- Cross curve (log10 of problem-mean of per-problem medians), trials 25/50/75/100% | time 50/75/100%:
  ESRS +0.604 +0.269 +0.119 +0.021 | +0.050 0.000 0.000; BO +0.477 −0.000 −0.097 −0.165 | −0.025 −0.116 −0.169; ESBO-GP +0.506 +0.024 −0.122 −0.204 | −0.125 −0.207 −0.273; ESBO-T-TabPFN +0.482 +0.056 −0.045 −0.177 | −0.076 −0.150 −0.276; DecPFN +0.504 +0.136 +0.014 −0.102 | −0.073 −0.210 −0.311; DecPFN-H +0.463 +0.139 −0.042 −0.124 | −0.138 −0.222 −0.279; XDecPFN-PI +0.037 −0.391 −0.511 −0.666 | −0.517 −0.618 −0.735; FT-PFN-EI −0.136 −0.453 −0.550 −0.593 | −0.539 −0.644 −0.712; ESBO-Oracle +0.459 −0.009 −0.080 −0.175 | −0.121 −0.175 −0.245.
  Thesis: groups at 100% trials 0.02 / −0.10 to −0.20 / −0.59, −0.67; FT-PFN-EI leads until ≈83% (crossover 83.3%); BO −0.17 vs −0.25 (Oracle) on time; improvement trials→time ESRS 0.021, BO 0.004, smallest other 0.069; Oracle ≈ ESBO-GP; Oracle worst ESBO-derived on Three-Tank (−0.159 vs ESBO-GP −0.261, T-TabPFN −0.232, DecPFN −0.459, DecPFN-H −0.682).
- Per-problem at 100% time (CL §4): BO −0.271/−0.155/−0.103/−0.439/+0.001; ESBO-GP −0.258/−0.261/−0.425/−0.432/−0.086; T-TabPFN −0.480/−0.232/−0.308/−0.402/−0.075; DecPFN −0.538/−0.459/−0.253/−0.461/−0.046; DecPFN-H −0.086/−0.682/−0.478/−0.396/−0.062; XDec-PI −0.752/−0.596/−0.674/−1.060/−0.719; FT-EI −0.760/−0.740/−0.789/−0.832/−0.517; Oracle −0.271/−0.159/−0.383/−0.480/−0.059 (boiler/threetank/pt2/cartpole/furuta).
- Pooled violin 50% / 100% (floor count at 100%): XDec-PI −0.619/−0.824 (14); FT-EI −0.598/−0.758 (12); DecPFN −0.074/−0.253 (6); Oracle −0.176/−0.313 (4); BO −0.002/−0.200 (2); DecPFN-H −0.087/−0.347 (2); ESBO-GP −0.126/−0.320 (1); T-TabPFN −0.057/−0.255 (0); ESRS 0/0 (0). Thesis quotes 14, 12, 6 at floor (SR:469-472). Display floor −4. No run unsolved.
- Crashes (counts at 100% sim time, SC:optplay_crashes; thesis SR:486-498): XDec-PI fewest everywhere, ≤ 3.8 (PT2 1.80, Cartpole 3.82, Furuta 2.33); Cartpole DecPFN-H 4.4 vs ESBO-GP 14.6; DecPFN/DecPFN-H PT2 2.30/2.65, Cartpole 4.55/4.40, Furuta 7.15/5.70; FT-EI 2.14/8.94/10.73; ESRS 11.9/53.8/15.7; BO 8.25/15.45/14.0; Oracle 6.50/16.45/5.90; T-TabPFN Furuta 6.55. Run counts: most 20, FT-EI 50/50/49, XDec-PI cartpole 50, furuta 43, T-TabPFN cartpole 50 (thesis silent).
- Timing (s per run, mean of 3 runs, 1 core; Boiler/Three-Tank/PT2/Cartpole/Furuta): BO 2.8/4.1/9.3/9.9/7.6; DecPFN 7.9/15.2/15.7/27.8/33.5; ESBO-GP 5.2/46.6/173.5/44.1/161.2; DecPFN-H 16.6/70.2/154.2/196.3/165.3; XDec-PI 19.5/83.8/119.5/140.0/142.4; FT-EI 31.3/101.5/135.5/183.6/354.2; T-TabPFN 123.0/1949.8/8946.4/2133.1/14546.0. DecPFN-H Transformer forecast 7.0/13.8/7.9/28.9/27.1, its Segment-GP forecast 4.8/46.9/131.3/150.1/122.7; Cartpole trials/run 148 (DecPFN-H) vs 74 (ESBO-GP), Segment-GP forecast 150 vs 34 s. Thesis: DecPFN-H ≈3× Boiler (17 vs 5), 4.5× Cartpole (196 vs 44); T-TabPFN >10× second-worst except Boiler.

### 7.6 Hardware (HR; HB; SC)
- Prediction benchmark median LL (without: SegGP / XDec / DecPFN | with: SegGP / XDec / DecPFN): n=0 —/−4.815/−3.698 | —/−5.095/−4.010; 1 −23.03/−3.376/−3.698 | −23.03/−3.625/−4.010; 2 −23.03/−2.998/−3.679 | −23.03/−3.312/−3.971; 5 −15.98/−2.804/−3.653 | −6.297/−3.117/−3.992; 11 −7.162/−2.651/−3.668 | −4.273/−2.937/−3.951; 22 −6.483/−2.516/−3.647 | −4.187/−2.985/−3.902; 45 −8.653/−2.389/−3.550 | −4.539/−2.796/−3.798; 90 −23.03/−0.031/−1.632 | −23.03/−0.809/−2.029. Claims: XDec n=0 beats all SegGP rungs without weight; with weight beats SegGP at every matching n; DecPFN beats SegGP everywhere.
- Ablation medians (SC, matches thesis): without weight dt+states −3.960, dt+no-states −3.867, no-dt+states −3.925, no-dt+no-states −4.013 (n=479); with weight −4.279, −4.129, −4.112, −4.293 (n=508). Thesis: spread 0.15 / 0.18; no-states better by 0.09 / 0.15; removing dead-time with states improves with weight by 0.17. [RF means −4.288 etc. are stale.]
- Crashed trials per seed (RF from HW): without weight ESBO-GP 6,9,13,17,8 (10.6); BO 14,30,25,17,22 (21.6); ESRS 7,16,15,12,14 (12.8); DecPFN-H 4,8,3,2,4 (4.2); DecPFN 5,5,5,3,2 (4.0); XDec 3,4,3,4,3 (3.4). With weight 7,6,8,16,9 (9.2); 8,5,8,20,12 (10.6); 4,8,5,6,12 (7.0); 2,3,5,5,3 (3.6); 4,5,6,4,3 (4.4); 3,5,2,5,3 (3.6). Thesis: BO nearly twice; PFN loops a third to a half of ESBO-GP.
- Regret per seed at 100% (log10, unscaled): see SC/RF; XDec gap to next median 0.68 without (0.692 − 0.008), 0.16 with (1.044 − 0.887); XDec with weight 0.79-1.09 + 1 floor seed, ESBO-GP 0.97-1.27; without weight XDec worst 0.19 beats all others' best except ESRS floor seed.
- Reset mean 9.17 s (3837 trials) / 9.68 s (4371); normalization 9.28 / 9.33 s. Acquisition wait: ESBO-GP 2.77/2.43, DecPFN-H 3.14/2.57, BO 0.20/0.20, DecPFN 0.14/0.14, XDec 0.07/0.07 s.
- Cost first 1000 s: plateau from ≈300 s (without), ≈900 s (with); XDec with weight reaches plateau ≈400 s. Medians in RF (e.g. with weight XDec 274.0/265.7/265.7/265.6/264.6 at 300/400/500/900/1000 s, ESBO-GP 350.0/310.2/310.2/276.0/268.0).
- Controllers at 500 s: without weight ESBO-GP 263.6 vs XDec 258.1 (swing-up ≈1 s); with weight ESBO-GP 310.2 vs XDec 265.7, ESBO-GP arm overshoots ≈45°, settles ≈2 s.

## 8. NAME MAP

| Thesis | Sim cell (optplay) | HW cell (hwplay) | Display in sidecars |
|---|---|---|---|
| ESRS | `Random uniform+cost` / `RandomESBO` (furuta) | `random_stenger` | ESRS |
| BO | `Plain GP` | `plain_gp` | BO |
| ESBO-GP | `Stenger ESBO` | `stenger` | ESBO-GP |
| ESBO-T-TabPFN | `ESBO-TabPFN` | — | ESBO-TabPFN / ESBO-T-TabPFN |
| DecPFN | `MTr 0.9` | `mtr0.99_k06binned` (RF: run as τ=0.9) | DecPFN-M-0.9-0.9 (sim), DecPFN-M-0.99-0.9 (hw) |
| DecPFN-H | `HMTr 0.9` | `htmr0.99_k06binned` | DecPFN-M-H-0.9-0.9 / -0.99-0.9 |
| XDecPFN-PI | `QCn PI abs 0.99` (`QC PI abs 0.99` cartpole/furuta) | `qc_pi_0.99` | XDecPFN-PI-0.99 / XDecPFN-PI-M-0.99 |
| FT-PFN-EI | `QF EI abs xgrid 0.99` | — | FT-PFN-EI-0.99 |
| ESBO-Oracle | `Oracle BO` | — | ESBO-Oracle |
| DecPFN thresholds | `MTr 0.99`, `HMTr 0.99`, `... ft0.99` | | DecPFN-M-τ-τcrash |

Models: DecPFN 1M binned `decpfn/epochs100/headbinned`; 100k `decpfn/epochs10/...`; heads `headbinned|headgaussian|headlinear`; DecPFN-no-St `.../states0` (sidecar `decpfn/nostates/...`); LC-PFN `lcpfn/epochs100/headbinned` → in final figures `lcpfn_zscore/epochs{10,100}/headbinned`; XDecPFN `dynformer_c1r2/cellnmatched` (also `celllnjw128` in context ladder, `cellf_s20260931` timing twin; ckpt `c1r2/cartpole_lnJ_W128_qw05/.../f_s20260931_seg2_final.pt`; `optplay.field.C1R2_LNJ_ROWS`); FT-PFN `dynformer_fr2` (code file `f3_model.py`); Trajectory-TabPFN `tabpfn/resolution{25,50,100,250}/typeraw`; Segment-TabPFN `stengertpfn/mode{rate,split}`; Segment-GP `stenger`. Problems: `boiler`, `threetank_pi`, `pt2_deadtime`, `cartpole`, `furuta_lqr_rigreset` (spec `furuta/lqr-new-realistic/rigreset`; old `furuta_lqr` not used). Fields: `<problem>@g0.05@bcapped`, timing `@timing`; predplay Oracle field; `ladder181_<problem>`. HW ablation arms `abl_{dt,nd}_{states,nostates}`, plants `deadtime-unif`, `nodeadtime`. E5 = 10 epochs = 100k, E6 = 100 epochs = 1M.

## 9. AMBIGUITIES (rebuild must decide)

1. Thesis vs RF numbers: rebuild targets thesis values (post z-score LC-PFN, new med_D)? RF/CL tables for prediction are stale. Which LC-PFN is canonical: z-scored inputs (thesis A:332) with hyperparameters from the old-normalization HPO?
2. Prediction timing device: thesis says workstation Quadro RTX 4000, timing bundle/sidecars say RTX 6000 (PR says it is the 4000). HW benchmark sidecar says times measured on CPU, caption says workstation GPU.
3. What exactly is "1M examples": 10⁶ unique trajectories (thesis table) or 100 epochs over a smaller pool (RF "E6 = 100 epochs")? Pool size and seeding (`generate_pool` was time-seeded)?
4. Number of attention heads for LC-PFN/DecPFN: d_k=d_v=16 fixed so N_head = d_model/16? Not stated for final configs.
5. Warmup 25% for all four models (A:354 phrasing) or only LC-PFN? Batch size for final LC-PFN/DecPFN trainings = 128? Epoch/step counts not given.
6. Validation set for HPO (size, how drawn): "20%-99%" cutoffs; SC mentions a 1024-curve validation set.
7. DecPFN input columns: thesis lists 4 Furuta states (2 levels + 2 rates → 7 inputs), deployed checkpoint has 13 input columns. Which channels exactly (motor voltage? cos/sin?) per problem?
8. XDecPFN: N_reuse = 256 (thesis) vs `W128` in the cell name; `qw05` meaning (query weight 0.5?); prior details in code (pilot crash rate, clamping, groups A/B in `c1r2_data.py`) not in thesis. Which XDecPFN cell is canonical (`cellnmatched` vs `celllnjw128` used in the context ladder on Cartpole/Furuta)?
9. Matched rule implementation: sample vs Gaussian moment, decision frequency (every step vs every ceil(T_max/30) steps), whether incumbent crash mass is kept, and the separate τ_crash (cells "τ-τ_crash"). Thesis defines only τ. Is the DecPFN-H variance comparison on the same distribution?
10. Hardware DecPFN cells are named `0.99`; RF says run as τ=0.9, sidecars label them DecPFN-M-0.99-0.9. Which τ was actually used on the rig?
11. Pure loop EI/PI on XDecPFN's ln J bins: closed form in raw J (thesis) vs log-space EI in code. How is J⁻_n clamped in ln-space?
12. FT-PFN: extra last-timestep token for every trial (code) or only truncated ones (thesis)? FT-PFN stopping decisions at XDecPFN token positions—what about the `T−1` cutoffs listed for the prediction benchmark?
13. ESBO-GP port: constant mean vs Stenger's zero mean on normalized outputs, length-scale bounds, Adam settings (lr, iterations, restarts), MES details (number of max-value samples, acquisition optimizer). GP_COLD_REFIT_GROWTH = 0.05 (code, sim field `g0.05`) is undocumented in the thesis: keep it? Exact cartpole tree with 0.0 exists separately.
14. ESBO-Oracle: thesis = ESBO-GP with point-mass true total cost as virtual obs; sidecar describes it as "perfect knowledge of whether to abort". Confirm implementation.
15. Initial design: uniform random (thesis) — Sobol or iid? Seeds for the 20 instances and per-run seeds unspecified; HW seeds 0-4 shared initial design across techniques.
16. Noise model: one formula for all problems (thesis) vs single-channel problems drawn differently in code (PR); Furuta motor-voltage noise is actuation noise. Which channels get noise in Three-Tank (only V4) — consistent?
17. Prior "Scale" factors: thesis gives supports; nominal defaults p0 are implied (e.g. Boiler heating 1.0) but not listed; tied tolerance "of order 1.5". Hardware ablation used `deadtime-unif`, deployed model `deadtime-strat` (stratified dead-time?) — which prior is canonical for Furuta?
18. Rejection rule for unsuccessful single trajectories: ρ_s̄ estimated how (per problem pilot, per instance)? Same for ρ_c, ρ_{s̄|c̄} in the set prior.
19. Furuta success: thesis = mean |φ_pend| < 5° over final 2 s; hwplay `successful` = no crash and final sample within 20° (used for HW incumbents/regret). Which defines the incumbent in sim and on hardware?
20. HW budget: thesis min{d,4} → 180 / 36000 matches the manifest, but the manifest records `d_budget=5` with explicit overrides; rebuild should hard-code 180/36000.
21. Context ladder: thesis says final trial of N_max-trial runs; SC says 181-trial runs (query index 180) with PT2 rungs up to 134; top rung labelled 179 but scored at 180.
22. Prediction benchmark seeds/phi: 50 phi per problem with varying seed counts (4/4/7/8/10); crashed-query matching (K_CRASH_PER_CELL=50) only for context-aware models; context-blind pooled over full field (858/1592 etc.). Thesis says "exactly 50 non-crashed and 50 crashed per (problem, n)".
23. Heads figure: centered on per-problem mean (code) but caption says median.
24. Simulated crash counts: some cells ran on 43-50 instances instead of 20 (thesis silent). Rebuild: 20 each?
25. TabPFN-3 exact package version/checkpoint file and device; Trajectory-TabPFN treatment of crashed/early-stopped trials in context beyond the formula.
26. Candidate generation: thesis 0.10 × extent std around top-5 completed controllers, 400+1600; is the same candidate set used for the GP-based MES optimizer, or only for pure loops?
27. Degeneracy screen threshold in landscape figure differs (filter 30 θ × 3 repeats, threshold 0.08) from the 100/25/10 screen; which for figure plants?
