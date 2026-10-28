# 06 — Decisions (Tony, 2026-10-28)

Everything in the chat question list of 2026-10-28 is accepted as recommended, except where this
file says otherwise. Later changes are appended at the end with a date.

## A. Repository
- A1. Repo `esbo-pfn`, package `esbopfn`, CLI `pfn`.
- A2. Fresh `git init` in `new/`, `/new/` ignored by the parent repo.
- A3. MIT. Code derived from other projects keeps its license (BlueRiverTech driver). **LC-PFN is
  rewritten from scratch, nothing vendored.** All four own trainers are written in the same short
  style on one shared training core.
- A4. Python `>=3.12,<3.13`.
- A5. YAML configs.

## B. Outputs and cluster
- B1. One root `PFN_HOME`.
- B2. No budget gate. `pfn submit` calls `sbatch`.
- B3. **No login-node GPU support.** Training runs as proper Slurm GPU jobs or locally.
- B4/B7. **No GitHub releases.** Checkpoints and benchmark results live in `$PFN_HOME` (models/,
  benchmarks/). Small derived result files and manifests may be committed under `thesis_artifacts/`.
  Large raw data (rig logs, checkpoints) stays outside git and is imported with `pfn import`.
- B5. CMake, MKL on the cluster, OpenBLAS or Accelerate elsewhere.
- B6. **All binaries are built on the target system.** No binaries in git.

## C. Hardware
- C1. Workstation not reachable now. Implement everything possible (client, runner, prediction
  benchmark, reset logic, simulated server for tests). Driver vendoring deferred.
- C2. Server shipped with `--plant sim` mode.
- C3. **Rig configurations are an open registry** (`configs/rigs.yaml`): `without_weight`,
  `with_weight` are two entries, any further modification is a new entry. Runs are per rig config.
- C4. Keep the hardware prediction ablation (prior flag `dead_time: false`, no-states models).

## D. Fidelity: the thesis wins everywhere
- D1–D4, D6–D17 as recommended (thesis).
- D2. Batch 128 and heads = d_model/16 confirmed.
- D3. Grids follow the thesis formulas. **Strides are not fixed per problem but derived**: the
  reference is cartpole with stride 15 (N_ref = ceil(1000/15) = 67), every other problem takes the
  stride whose N_grid = ceil(T/stride) is closest to N_ref (ties → smaller stride). This reproduces
  the thesis table (11/7/8/15/9).
- D5. **XDecPFN and FT-PFN follow the thesis text** (linear embeddings, z-scored time, AdamW with
  clip 1, plain mean loss, no readout skip, no unrun-query weight).
- D11. `gp_cold_refit_growth = 0.05` kept as an explicit parameter (see 07 §"open points").

## E. Benchmarks are not run during the build
- Old thesis data is imported and reused for README and figure checks. The code must be able to
  run everything, nothing is executed beyond tests.

## F. Cluster access
- ssh `dln18500@login23-1.hpc.itc.rwth-aachen.de`, checkout at `bachelorarbeit/impl`. Needs an
  interactive login (ControlMaster socket), see chat. Workstation unavailable.

## G. Scope
- G1–G3, G5 as recommended.
- G4. **No HPO for XDecPFN and FT-PFN.** Their hyperparameters are fixed in `configs/hparams/`.
  HPO pipeline only for LC-PFN/DecPFN (and used by lcpfn-last / decpfn-nostates through inheritance).
- G6. Data generation for every model including FT-PFN Furuta is a runnable command, nothing is
  "artifact only".

## Amendments
- 2026-10-28 (Tony): **D5 amended.** XDecPFN and FT-PFN follow the thesis text in everything except
  gradient clipping, which stays OFF for these two models (no grad-norm clip). LC-PFN and DecPFN keep
  clip 1 as the thesis states. This is a documented deviation (07 §11, docs/DEVIATIONS.md).
