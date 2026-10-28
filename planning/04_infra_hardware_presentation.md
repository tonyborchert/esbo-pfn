# 04 — Infrastructure, hardware, assets, presentation figure

Audit of everything *around* the scientific code (HPC, Slurm, rig, checkpoints, plotting style, tests, git,
local machine). Read-only survey, 2026-10-28. Paths are relative to `bachelorarbeit/` unless absolute.
`impl/` = `/Users/tonyborchert/Documents/Uni/bachelorarbeit/impl`.

Three machines appear in the code. Keep them apart in the new repo:

| machine | identity in code | role |
|---|---|---|
| RWTH CLAIX-2023 (HPC) | user `dln18500`, `$HOME=/rwthfs/rz/cluster/home/dln18500`, `$HPCWORK=/hpcwork/dln18500`, `$WORK=/work/dln18500`, login node `login23-g-1` | data generation, training, simulated optimization benchmark (optplay) |
| Lab workstation (Linux) | `/home/tony/bachelorarbeit/impl/...` (`impl/prediction/hwplay/README.md:91`, `impl/prediction/experiments/furuta/hardware_demo.ipynb` cells `FURUTA_PYTHON`/`FURUTA_SERVER`) | Intel Xeon W-1290P + Quadro RTX 4000 8 GB (`writing/thesis/sections/appendix.tex:411`). Drives the QUBE rig, runs hwplay, the prediction benchmark (predplay) and the timing bundle |
| MacBook (this machine) | `/Users/tonyborchert/Documents/Uni/bachelorarbeit` | writing, presentation, figures. Apple M5, 24 GB, arm64 |

---

## 1. HPC environment (RWTH CLAIX-2023)

Primary sources: `impl/dynformer/ENVIRONMENT.md` (430 lines, verified 2026-08-17..09-20) and
`impl/prediction/CLAUDE.md:17-115`.

### 1.1 Modules

| purpose | exact line | source |
|---|---|---|
| Python (always, first) | `module purge && module load GCCcore/13.3.0 Python/3.12.3` | `ENVIRONMENT.md:20`, `prediction/CLAUDE.md:25` |
| C++ build (furuta, needs MKL) | `module purge && module load GCC/13.3.0 imkl/2024.2.0` | `ENVIRONMENT.md:29`, `prediction/CLAUDE.md:66` |
| C++ build (cartpole, plants, no MKL) | `module load GCC/13.3.0` | `prediction/CLAUDE.md:65` |
| CUDA (only for nvcc, not for torch) | `module load CUDA/12.4.0` (also `CUDA/12.8.0` exists) | `ENVIRONMENT.md:32-40` |
| TeX for matplotlib usetex (Charter) | `module load GCCcore/13.3.0 Python/3.12.3 texlive/20240312` | `ENVIRONMENT.md:228-231` |

Traps worth carrying into docs:
- Without the Python module, `import numpy` SIGSEGVs (exit 139) on the login node (`ENVIRONMENT.md:23`).
- An unknown module name aborts the whole `module load` line. `cuda/12.4` does not exist. **`impl/prediction/manager/base.slurm:21` still says `module load cuda/12.4`** (a latent bug, harmless only because it is on its own line and torch's cu128 wheel brings its own runtime).
- Module load and python must be in the same shell invocation (`prediction/CLAUDE.md:28-35`).
- C++ binaries are built `-static-libstdc++ -static-libgcc` + MKL rpath, so they run under `module purge` (`prediction/CLAUDE.md:69-70`).
- `torch.cuda.is_available()` poisons later `fork()`. Use `torch.cuda.device_count() > 0` pre-fork (`ENVIRONMENT.md:296-301`).
- matplotlib usetex fails with the local TeX Live 2020 (missing `mathdesign.sty`, Charter metrics). `OPTIM_DISABLE_USETEX=1` falls back to mathtext (`ENVIRONMENT.md:198-239`, `prediction/manager/base.slurm:16`).

### 1.2 Accounts, partitions, quotas

| account | use | partitions | notes |
|---|---|---|---|
| `rwth2055` | **all CPU jobs** | `c23ms` (also `c23mm`, `c23ml`) | 360k core-h total, max 100 h wall, max 24 cores/job. No GPU association (`ENVIRONMENT.md:109-110,147`) |
| `thes2287` (= `$PROJECT`) | **GPU jobs only** | `c23g` (H100) | 1,500 GPU-h period, CPU side over quota and demoted to `_low` (`ENVIRONMENT.md:111-112,141-159`) |

`c23g` node: 96 CPUs, 4x H100, 488 GB. 1 GPU bills as 24 core-h-equivalent, covering 24 CPUs and ~122 GB
(`ENVIRONMENT.md:161-171`). Thesis appendix states "H100 with 8 cores of Xeon Platinum 8468" for training
(`writing/thesis/sections/appendix.tex:409`).

Counted over all 171 `*.slurm` files in `impl/` (excl. `.venv`): `--account=rwth2055` 70x, `--account=thes2287` 5x,
`--partition=c23ms` 94x, `c23g` 8x, `--gres=gpu:1` 7x (one `gpu:4`). Most scripts do not hardcode `-A`, it is passed
on the `slurm_submit.sh` command line.

### 1.3 Login-node limits

- **CPU:** user cgroup capped at 4 cores (`cpu.max = 400000 100000`), `nproc` reports 1 (`ENVIRONMENT.md:61-68`, `prediction/CLAUDE.md:92-94`).
- **GPU:** 4x H100 95 GB usable for short runs. **Processes on a login GPU are killed after 30 min** (`ENVIRONMENT.md:51-53`). A 20-cell sweep was SIGTERMed at ~33 min (`prediction/CLAUDE.md:96-99`). Thesis training scripts work around this by resuming from `ckpt_last.pt` (`impl/dynformer/experiments/fr2/fr2_run_gpu0.sh:12-13`).
- **GPU UUID quirk:** physical GPU 2 is MIG-partitioned into 7x `1g.12gb`. CUDA ordinals do not match `nvidia-smi` indices (`CUDA_VISIBLE_DEVICES=3` lands on a MIG slice, 4-9x slower, no error). Always select by UUID (`ENVIRONMENT.md:55-59`, table in `prediction/CLAUDE.md:72-86`). Hardcoded UUIDs appear across dynformer (`GPU-89dc5e46-…` 37x, `GPU-0911978f-…` 33x, `GPU-8ff8d0c7-…` 24x, e.g. `fr2_run_gpu0.sh:24`, `experiments/s12/s12_data_tests.py:8`). A login-node GPU queue runner exists in `impl/dynformer/experiments/recipe/driver/` (`gpus.py` busy classifier, `run_queue.py`, `queue.yaml`).
- Short trainings on the login GPU were often faster than `c23g` because of queue wait (~10 h estimated start) (`prediction/CLAUDE.md:88-95`). `training_time` is not comparable across sessions/machines (`prediction/CLAUDE.md:101-111`).

### 1.4 Typical resource requests per job type

| job type | example | header / CLI | engine |
|---|---|---|---|
| C++ pool generation (CPU) | `impl/prediction/tools/furuta_cpp/gen.slurm:12-19` | `--cpus-per-task=48 --mem=16G -p c23ms --time=04:00:00 -A rwth2055` | `srun furuta_gen gen …`, `MKL_NUM_THREADS=1` |
| Python pool / prior data (CPU) | `impl/dynformer/experiments/recipe/pools/pool_cartpole_train.slurm`, `recipe/fdata/cartpole_f_gen.slurm`, `experiments/c1r2/c1r2_gen.slurm` | `--cpus-per-task=24 --time=00:45:00..03:00:00` (24 = rwth2055 per-job max) | `uv run --no-sync --project ../prediction python …` |
| Simulated optimization benchmark (CPU) | `impl/dynformer/optplay/slurm/run_problem.slurm`, `run_qx.slurm` (array, one task per instance `0-19`) | `-p c23ms --time=00:30:00..01:00:00 --cpus-per-task=4..8`, `--export=ALL,PROBLEM=…,WORKERS=…,SKIP_POST=1`, then `post_problem.slurm --dependency=afterok:<id>` | `python -m optplay.run` |
| Prediction scoring (CPU/GPU) | `impl/dynformer/predplay/slurm/predict.slurm`, `predict_gpu.slurm:4-7` | GPU: `-A $PROJECT -p c23g --gres=gpu:1 --time=02:00:00 --cpus-per-task=8 --mem=64G` | `python -m predplay.predict --shard …` |
| PFN training via manager (GPU) | `impl/prediction/manager/load.py:205-226` builds `sbatch` → `manager/base.slurm` | `--partition=c23g --gres=gpu:1 --cpus-per-task=num_workers+1 --mem=32G` (or auto-sized), `--time` estimated from gen speed ×1.5 (`load.py:13-15,42,190-201`) | `uv run --no-sync ./manager/load.py "$@"` |
| Thesis XDecPFN/FT-PFN training | `impl/dynformer/experiments/fr2/fr2_run_gpu0.sh`, recipe driver | **login-node H100 by UUID**, nohup, resume loop, not Slurm | `fr2_train.py`, `c1r2_*` |

Frequencies across all scripts: `--cpus-per-task=24` 79x, `8` 20x, `4` 11x. `--mem=16G` 36x, `32G` 19x, `8G` 14x,
up to `180G`. `--time` mostly 01:00–08:00, six at 24:00:00. 6 array scripts (`--array=0-7 … 0-116`), 14 uses of
`SLURM_ARRAY_TASK_ID`, `srun` used in 21 places.

### 1.5 The budget gate `slurm_submit.sh`

`impl/dynformer/scripts/slurm_submit.sh` wraps `sbatch` 1:1, parses CLI + `#SBATCH` lines (CLI wins, `:56-119`),
refuses jobs without `--time` (`:135`), above 5 GPU-h or 100 core-h per job (`:11-12,29-30`), and keeps per-run
ledgers `logs/slurm_budget/<run-id>.tsv` (`:17-20`). Env knobs `DYNFORMER_SLURM_MAX_GPU_H`, `DYNFORMER_SLURM_MAX_CPU_H`,
`DYNFORMER_SLURM_RUN_{GPU,CPU}_H`, `DYNFORMER_RUN_ID`. This was an agent-safety device. For a public repo it is optional.

### 1.6 Storage locations

Env-based roots, with their defaults:

| variable / root | default | defined at | holds |
|---|---|---|---|
| `DATA` | `$HPCWORK/datasets` (else `/tmp/datasets`) | `impl/prediction/manager/storage.py:8-9` | prediction pools `<DATA>/<dataset>/pool.npy_raw` (`manager/register.py:81-82`) |
| `MODELS` | `models` (relative to cwd = `impl/prediction`) | `manager/storage.py:5-6` | trained predictors `MODELS/<predictor>/<dataset>/<n>/model.pt` (`register.py:72-77,134-156`) |
| `OPTPLAY_CHECKPOINTS` | `$HPCWORK/optplay` else `optplay/checkpoints` | `impl/dynformer/optplay/optplay/env.py:28-33` | raw optplay runs (`<field>/phi_*/runs/*.json`, 29 GB-class) |
| (optplay results/figures) | `optplay/results`, `optplay/figures`, `optplay/instances` (in repo) | `optplay/env.py:35-37` | aggregated JSON/CSV, figures |
| `PREDPLAY_DATA` | `$HPCWORK/predplay` (`HPCWORK` defaults to `/hpcwork/dln18500`) | `impl/dynformer/predplay/predplay/env.py:29-33` | predplay preds `preds/<problem>/<model_id>…`, fields |
| `HPCWORK` hard default | `/hpcwork/dln18500` | `impl/dynformer/optplay/optplay/model_specs.py:93` | all XDecPFN/FT-PFN checkpoints below |
| `HWPLAY_RUNS` | `impl/prediction/hwplay/runs` | `impl/prediction/hwplay/config.py:138-140` | rig `trials.jsonl` per cell/seed |
| TabPFN checkpoints | `impl/prediction/predictors/tpfn/models/` | `predictors/tpfn/__init__.py:34`, `optimizers/tabpfn_surrogate.py:61` | `tabpfn-v3-regressor-v3_default.ckpt` etc. (§4) |
| uv lock | `$HOME/.cache/uv_sync.lock` | `manager/base.slurm:41-42` | flock to serialise `uv sync` |

Checkpoint and data trees under `$HPCWORK` (from slurm/py grep, thesis-relevant ones first):
- `$HPCWORK/c1r2/<problem>_lnJ_W128_qw05[_nmatched]/ckpt_final/…` — **XDecPFN** checkpoints used by optplay (`optplay/optplay/field.py:311-324,409-415`), plus `c1r2/<problem>/{tranche,heldout,ckpt,phi}` (training data).
- `/work/dln18500/c1r2/furuta_lnJ_W128_qw05_successgate/ckpt_final/` — XDecPFN Furuta, **on `$WORK`, not `$HPCWORK`** (`field.py:324`).
- `$HPCWORK/fr2_F/runs/<key>_3500000/final.pt` — **FT-PFN** (dynformer_fr2) checkpoints (`optplay/optplay/fr2_predictor.py:13,103`). Furuta one at `/work/dln18500/fr2_F/runs/furuta180sg_3500000` (`field.py:348`, `predplay/predplay/registry.py:602`).
- `$HPCWORK/recipe/<problem>/{c1pool/{train,heldout},f/shards,c1/ckpt,f/ckpt,calib.json,z_stats.json}` — recipe pools and checkpoints (`optplay/model_specs.py:58-59`, `experiments/recipe/pools/*.slurm`).
- `$HPCWORK/optplay`, `$HPCWORK/optplay_{i11b,i11c,s51,s51b_cp}` — optplay raw runs (current + historic campaigns).
- `$HPCWORK/predplay/{preds,field}` — predplay outputs.
- `$HPCWORK/oraclebench/…` (`model_specs.py:191`), `$HPCWORK/datasets/…` (DATA default).
- Historic: `s12, s13, s50, i1, i5, i11b, i11c, f8_field, f25_oracle_bench, f26, f31, f37, b59, pt64, pt135, fur128, fur225, curves_shard` (pools, ckpts, scores of exploration phases).

Logs: mostly `logs/%x_%j.out` relative to `SLURM_SUBMIT_DIR` (48x), some absolute
`/rwthfs/rz/cluster/home/dln18500/bachelorarbeit/impl/dynformer/logs/...` (e.g. `experiments/c1r2/c1r2_gen.slurm`, `fr2` scripts).

### 1.7 `uv` and Python versions

- Only **one** uv project serves both codebases: `impl/prediction/pyproject.toml`. `impl/dynformer` has no pyproject. It always runs `uv run --no-sync --project ../prediction …` (`ENVIRONMENT.md:278-281`). Do not put `prediction/util` on `PYTHONPATH` first, `util/math.py` shadows stdlib `math` (`ENVIRONMENT.md:283`, `base.slurm:29-31`).
- `uv run --no-sync` always. `uv` is not on a batch job's PATH, so every script does `export PATH="$HOME/.local/bin:$PATH"` (116 occurrences, `base.slurm:8`). Concurrent `uv sync` tore `.venv` down twice, fix: `flock "$HOME/.cache/uv_sync.lock" uv sync --frozen --no-dev` (`base.slurm:34-42`).
- `impl/prediction/pyproject.toml`: `requires-python = ">=3.12,<3.14"` (`:4`), `.python-version` = `3.12`. HPC venv runs on module Python 3.12.3. Mac venv `impl/prediction/.venv` uses uv-managed CPython 3.12 (`bachelorarbeit/.uv/python/cpython-3.12-macos-aarch64-none`).
- torch: `cu128` index on Linux (`pyproject.toml:43-51`, lock `torch 2.11.0+cu128`), PyPI arm64/MPS wheel on macOS (`required-environments`, `:37`, lock `torch 2.12.1`).
- `tabpfn>=8.0.5` (lock 8.1.0), `tabpfn-client>=0.3.2` (0.3.3) (`:27-28`).
- `gym_brt` is a dependency via `[tool.uv.sources] gym_brt = { path = "../furuta/furuta/quanser-driver", editable = true }` (`:30,48`). **That path does not exist on the Mac or in git** (only on the lab workstation), so `uv sync` on a fresh checkout fails. It is only needed for `render3d.py`'s MuJoCo assets (`experiments/furuta/CLAUDE.md:762-770`).
- `impl/furuta/pyproject.toml`: name `vision-based-furuta-pendulum-master`, `requires-python >=3.11`, Mac `.venv` is **CPython 3.14** (`impl/furuta/.venv/pyvenv.cfg`). Irrelevant to the thesis, the real rig server runs in a **Python 3.8** venv on the workstation (`/home/tony/bachelorarbeit/impl/furuta/venv/bin/python3`).

---

## 2. Slurm patterns (catalogue)

Common skeleton (e.g. `optplay/slurm/run_problem.slurm`, `predplay/slurm/predict_gpu.slurm`, `recipe/pools/*.slurm`):

```bash
#!/usr/bin/env bash
# usage block: the exact slurm_submit.sh line incl. -A/-p/--time/--cpus-per-task/--export
#SBATCH --job-name=… --output=<dir>/logs/%x-%j.out --error=…   # sometimes nodes/ntasks/cpus/mem/time/partition
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
module purge && module load GCCcore/13.3.0 Python/3.12.3
export OPTIM_DISABLE_USETEX=1  [PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True]  [OMP_NUM_THREADS=1]
cd "${SLURM_SUBMIT_DIR}/<subproject>"   # or a hardcoded ROOT=/rwthfs/.../dynformer
: "${PROBLEM:?set PROBLEM}"; WORKERS="${WORKERS:-…}"     # arguments arrive as --export env vars
ARGS=(…); uv run --no-sync --project ../prediction python -m <pkg>.<module> "${ARGS[@]}"
```

Variants:

| pattern | where | argument passing | notes |
|---|---|---|---|
| **manager** | `prediction/manager/base.slurm` + `manager/load.py --slurm` | `"$@"` forwarded to `load.py` (sbatch extra args after the script) | `USE_CUDA` env decides CUDA path. Does `flock uv sync` (only script that syncs). load.py imports numpy on the login node before submitting (`prediction/CLAUDE.md:57-60`) |
| **C++ generator** | `prediction/tools/{furuta,cartpole}_cpp/{gen,validate}.slurm` | `"$@"` → `furuta_gen gen`, `--workers` defaults to `SLURM_CPUS_PER_TASK` | `make -C tools/furuta_cpp` inside the job, `srun`, `MKL_NUM_THREADS=1` |
| **env-var driver** | `dynformer/optplay/slurm/*.slurm`, `predplay/slurm/*.slurm`, `experiments/*/*.slurm` | `--export=ALL,PROBLEM=…,WORKERS=…,INSTANCES=…,SKIP_POST=1` | `:?` guards on required vars |
| **array** | `optplay/slurm/run_qx.slurm`, `landscape_visualization/run_*.slurm`, `handoff_matched_live_prediction/scripts/submit_array.slurm` | `INSTANCES=${INSTANCES:-$SLURM_ARRAY_TASK_ID}` | followed by a `--dependency=afterok` post job |
| **login-node GPU loop** | `dynformer/experiments/fr2/fr2_run_gpu0.sh`, `recipe/driver/start.sh` | env knobs (`FR2_GPU_UUID`, `FR2_CELLS`, …) | not Slurm. nohup + resume on the 30-min kill |

Hardcoded things to parameterise in the new repo:
- user/home paths: `/rwthfs/rz/cluster/home/dln18500/bachelorarbeit/impl/dynformer` (30 occurrences in slurm, plus `ROOT=` in `.sh`), `/hpcwork/dln18500/...`, `/work/dln18500/...` (in `.py`: `optplay/field.py:311-324,348,409-415`, `predplay/registry.py:602`, `model_specs.py:93`).
- account/partition (`rwth2055`, `thes2287`, `c23ms`, `c23g`) and the module list.
- GPU UUIDs.
- relative log dirs that assume a submit dir (`logs/`, `optplay/slurm/logs/`, `predplay/slurm/`).
- `base.slurm`'s invalid `cuda/12.4` (`:21`).

Proposal for ONE generic layer: a `cluster.toml` (or env file) with `account_cpu`, `account_gpu`, `partition_cpu`,
`partition_gpu`, `modules`, `pre_commands` (PATH, env), `storage_root`, `log_dir`. A single `submit.py` renders a
job script from a template (CPU/GPU/array), passes the experiment command and its args directly (no `--export`
soup), and supports `--local` to run the same command without Slurm (essential for Mac users and the CI).

---

## 3. Hardware (Quanser QUBE-Servo 2 Furuta pendulum)

### 3.1 `impl/furuta` on the Mac is a husk

Contents (git-tracked, 16 files): `main.py` (a "Hello from vision-based-furuta-pendulum-master!" stub),
`pyproject.toml`, `uv.lock`, `gym_brt/quanser/quanser_wrapper/quanser_wrapper.c` (29,782-line Cython 0.29.21
output), three **empty** `__init__.py` (`gym_brt/blackfly`, `gym_brt/envs/simulation`, `gym_brt/quanser/quanser_wrapper`),
`build/temp.linux-x86_64-3.{6,7,8}/…/quanser_wrapper.o`, `bad_scripts/*.py` (use `QubeSwingupEnv`, `QubeFlipUpControl`),
`bad_scripts/recording.mp4` (48 MB), `tests/monitor/*.mp4`. Git history: `a0e2f51 initial commit`, `a55218b dsme institute 31.08`.

- It is a copy of **BlueRiverTech's `quanser-openai-driver`** (package `gym_brt`), in a "vision-based furuta pendulum" variant (the `blackfly` camera package, `opencv`, `mss`). The Python sources (`envs`, `control`, `quanser_wrapper.pyx`, `setup.py`) are **not** here.
- The working driver lives only on the workstation: `impl/furuta/furuta/quanser-driver/` with `remote_env_server.py` and a Python-3.8 venv (`impl/prediction/experiments/furuta/CLAUDE.md:23-30`, `hardware_demo.ipynb`).
- C extension: Cython-generated, includes `/opt/quanser/hil_sdk/include/{hil.h,quanser_types.h}`, links `hil quanser_runtime quanser_common rt pthread dl m c` from `/opt/quanser/hil_sdk/lib` (`quanser_wrapper.c:5-33,647-648`). Generated originally in an Anaconda py3.7 env (`quanser_wrapper.c:996`). Linux-only (`rt`, `USBDEVFS_RESET` in `reset_qube.py:24,40`, `/proc`, `dmesg`). `setup.py` skips the extension if Cython/HIL SDK are missing (`experiments/furuta/CLAUDE.md:768-770`).

### 3.2 How the thesis code talks to the rig

`impl/prediction/experiments/furuta/hardware.py` (529 lines):
- `RemoteFurutaEnv` (`:146`) opens `ws://localhost:8765` (`:163,185`) to `remote_env_server.py`, which round-trips `reset_down` / `step(voltage)` / `center` / `close`. No controller logic on the server side.
- `ping_interval=None` is required (server blocks for seconds in HIL calls, server pins `websockets 13.1`, client `17.0.1`). Per-command reply watchdog `REPLY_TIMEOUTS = {step 15, reset_down 90, center 60, close 30}` s (`:160`). On timeout the socket is discarded (`experiments/furuta/CLAUDE.md:31-55`).
- `FurutaHardwareSystem(FurutaSwingupSystem)` (`:310`) reuses the sim's controller/cost/safety via the `System._make_env` hook. `phi` has no effect on hardware, only `phi_hat`. Pass `noise_stds=0`.
- `HARDWARE_THETA` (`:35ff`): rig-measured controller configs (e.g. `lqr_working` 15/15).
- Reset cycles in `fast_hardware.py` (1,484 lines): `FastFurutaHardwareSystem` (`:255`, `kick_voltage=0.6` `:286`), `StillStart…` (`:800`), `CentredTick…` (`:872`), `SettledStart…` (`:1093`), `SettledTick…` (`:1319`), `SettledTickCenter…` (`:1342`). Selected by `HWPLAY_CYCLE` (default `"tick"`, `hwplay/config.py:194-202`). Thesis trials record `recovery.kind` per trial, e.g. `"fast"` with `kick_voltage 0.6` and a signed kick in `runs_holdcost5_newbox_reset_fixed/lqr_hardware/stenger/seed0/trials.jsonl`.
- Simulated mirror of the rig reset: `experiments/furuta/tick_reset.py` (`KICK_VOLTAGE = 0.10` `:58`, `FurutaTickResetSystem` `:139`).

Rig operation, as documented (`hwplay/README.md:129-147`, `experiments/furuta/CLAUDE.md:210-232,713-745`):
1. Start `remote_env_server.py` with `python3 -u` from furuta's py3.8 venv. Check `pgrep -af remote_env_server` (two servers silently fight over port 8765).
2. Board allows one process. `Board could not be opened` = `QERR_BOARD_ALREADY_OPEN`. Recovery: `reset_qube.py` (USB reset, may need physical replug).
3. Opening the board zeroes the encoders wherever the arm is (no homing). Arm travel 257.2°, encoder drift one count (`measure_limits.py`, `check_repeatability.py`). The arm self-centres slightly (cable torque).
4. Re-run a known-good config first (`lqr_working` 15/15 on 08-17, 1/3 after a re-enumeration on 08-24).
5. Between trials, the reset of thesis Algorithm `alg:app:hw-reset` (`writing/thesis/sections/appendix.tex:986-1022`): brake, coast 1.5 s, PD-centre the arm (K_P 3.0, K_D 0.2, min 0.35 V), wait for rest 2 s, re-zero the arm, signed 0.6 V kick for 30 ms. Connection reset every 10th trial. Reset time replaced by a per-setup constant for the wall-clock axes.

Thesis hardware vocabulary:
- **"without weight" / "with weight"**: with weight = a 5 g tip mass at the end of the pendulum link (`writing/thesis/sections/setup.tex:462-467,415-431`). Directory map: `reset_fixed` = without weight, `reset_fixed_different_plant` = with weight (`impl/dynformer/thesis_figures/HANDOVER.md:89`, `thesis_figures/README.md:43`, `GRAPHSINEED.md:36-37`).
- **"newbox"**: the controller search box `THETA_BOX["furuta/lqr"]` changed on 2026-09-07 (mu ≤ 90, ref_energy 0.70..1.20, barrier_kp 1.5..3.5) and was made identical to the sim box (`hwplay/config.py:371-378`). Trees before that are refused by the field guard.
- **"holdcost5"**: objective `FurutaHoldCost(5deg)`, pendulum angle only, `|alpha|·(1+20(1-p))` (`hwplay/FINDINGS.md:552-556`, `experiments/furuta/objective.py`).
- Variant `lqr`: 120 Hz, `T=600`, `d=8`, budget `D_BUDGET=5` (`hwplay/config.py:72-93`). Thesis manifest: `trials_budget 180`, `t_budget 36000` steps, seeds 0–4 (`runs_holdcost5_newbox_reset_fixed_different_plant/lqr_hardware/manifest.json`).
- Rig timing: ~26 ms per step dominated by the websocket round trip (`experiments/furuta/CLAUDE.md:205-210`).

### 3.3 `impl/prediction/hwplay`

Package: `config.py` (402), `field.py` (504, `CELLS`), `run.py` (343, `run|list|status|pause|unpause`), `aggregate.py`, `figures.py`,
`prediction_benchmark.py`, `prediction_scatter.py`, `selfcheck.py` (418, proves resumability on the simulator), `README.md` (448, the manual),
`FINDINGS.md` (924). Driver is `experiments/furuta/hardware_benchmark.py` (1,640 lines, `TrialRecord`). Workflow `hwplay/README.md:89-118`
(`PYTHONPATH=.:util python -m hwplay.run run`, `--plant bridge` dry run via the websocket server's ODE sim).

Run trees (all git-tracked, 221 `trials.jsonl`):

| dir | size | status |
|---|---|---|
| `runs/` | 219 M | first field (`FurutaCost`), old box |
| `runs_dtunif20/` | 0.7 M | probe |
| `runs_holdcost5/` | 47 M | second field, old box |
| `runs_step30/` | 18 M | maneuver field (step to +30°) |
| `runs_holdcost5_newbox/` | 252 M | new box, before the reset fix. Used as source for `reset_replay` (`experiments/furuta/CLAUDE.md:568,592`) |
| `runs_holdcost5_newbox_different_pendulum/` | 168 M | superseded weight run |
| **`runs_holdcost5_newbox_reset_fixed/`** | 121 M | **thesis, without weight** |
| **`runs_holdcost5_newbox_reset_fixed_different_plant/`** | 114 M | **thesis, with weight** |

Thesis cells in both trees (`writing/thesis/data/results_facts.md:17-31`): `random_stenger` (ESRS), `plain_gp` (BO), `stenger` (ESBO-GP),
`mtr0.99_k06binned` (DecPFN, run as τ=0.9), `htmr0.99_k06binned` (DecPFN-H), `qc_pi_0.99` (XDecPFN-PI). The without-weight tree also has
`qc_pi_0.99_n10k` (not in the thesis name map). Derived CSVs are tracked too: `hwplay/results/trials_lqr_hardware*.csv` (10.8 MB, 7.5 MB).

`impl/prediction/furuta_rigreset_models/` contains only `MANIFEST.json` / `MANIFEST.md` (20 K). The 12 DecPFN `rigreset_*.pt`
checkpoints it lists (nodeadtime / deadtime-strat / deadtime-unif × kick 0.6/0.2 × binned/nostates, exported by
`scripts/export_rigreset_models.py`) are **not present on the Mac** (`*.pt` is gitignored, `impl/.gitignore:234`).
`experiments/furuta/reset_replay_bench.py:45` defaults to this dir. These are the DecPFN models the rig cells loaded via `HWPLAY_MODEL`.

### 3.4 `impl/cartpole_real` — not used by the thesis

Four files: `external_control_host.py`, `external_control_host_macos.py`, `keyboard_control.py`, `external_control_interface.md`
(a binary serial protocol to an STM32 Nucleo cart-pole at 921600 baud, 50 Hz state frames, ±9 V). No reference from `impl/prediction`
or `impl/dynformer`, and the thesis's only hardware is the QUBE (`setup.tex:450-480`, `hw_results.tex`). "Cartpole" in the thesis is
the simulated problem. **Exclude from the new repo.**

---

## 4. TabPFN checkpoints and other downloaded weights

| asset | how it is obtained | where expected | used by |
|---|---|---|---|
| **TabPFN v3** regressor (`tabpfn-v3-regressor-v3_default.ckpt`) | `tabpfn` package 8.1.0 (PyPI). Checkpoint files placed by hand. v2/v2.5/v2.6 names also mapped | `impl/prediction/predictors/tpfn/models/<file>` (`predictors/tpfn/__init__.py:34-41,52-57`, `optimizers/tabpfn_surrogate.py:61-66,219`). **Directory absent on the Mac, `*.ckpt` gitignored** (`impl/.gitignore:232`) | TabPFN predictor, Segment-TabPFN (`stengertpfn/__init__.py:76,178`), ESBO-TabPFN surrogate. predplay pins `TABPFN_VERSION = "3"` (`dynformer/predplay/predplay/registry.py:76`) |
| TabPFN hosted API | `tabpfn-client` 0.3.3, `client=True` option maps versions to `v3_default` etc. (`tpfn/__init__.py:59-67,166,183`) | network + token | optional, not the thesis path |
| device | `mps` → `cuda` → `cpu` (`tpfn/__init__.py:70-76`). `_PREDICT_CHUNK=64` to avoid MPS OOM (`:46`) | | |
| **LC-PFN** | **retrained by the user** (`predictors/lcpfn/custom.py`, model id `lcpfn/epochs100/headbinned`, `results_facts.md` model-id table) | `MODELS/lcpfn/...` | thesis LC-PFN |
| LC-PFN pretrained (Adriaensen et al.) | legacy vendored package `impl/prediction/training/lcpfn/lcpfn/__init__.py:12-35` downloads `pfn_EPOCH1000_EMSIZE512_NLAYERS{6,12}_NBUCKETS1000.pt.gz` from `ml.informatik.uni-freiburg.de/research-artifacts/lcpfn/` | package dir | not used by the thesis predictors |
| **FT-PFN** | **own model** "FT-PFN-shaped" (`dynformer_fr2`), trained in `dynformer/experiments/fr2` (`predplay/registry.py:544-563`). No ifBO checkpoint and no `ifbo` import anywhere | `$HPCWORK/fr2_F/runs/<key>_3500000/final.pt`, furuta on `/work` | FT-PFN-EI |
| **XDecPFN** | own (`dynformer_c1r2`, cell `nmatched`) | `$HPCWORK/c1r2/<problem>_lnJ_W128_qw05[_nmatched]/ckpt_final/` | XDecPFN-PI |
| **DecPFN** | own (`decpfn/epochs100/headbinned`) | `MODELS/decpfn/...`, rig versions in `furuta_rigreset_models/` | DecPFN, DecPFN-H |
| pfns4bo | `impl/pfns4bo_test/{test,better_test}.py` import `pfns4bo` (not in any pyproject). Scratch experiment | — | not used |
| Other HF weights | only literature notes (`dynformer/notes/01_literature/04_panda_…md:153`, `31_tabpfn_ts_…md:93`) | — | not used |

For the new repo: every checkpoint the thesis numbers depend on (TabPFN v3 ckpt, DecPFN/LC-PFN/XDecPFN/FT-PFN finals, 12 rigreset
DecPFNs) is outside git and on HPC or workstation disks. A `download_assets.py` + a manifest with sha256 hashes is needed. TabPFN v2.5+
weights are distributed under Prior Labs' license on Hugging Face. Check whether the v3 checkpoint may be redistributed or must be
fetched by the user (likely the latter, gated download).

---

## 5. Presentation 3D figure and plotting style

### 5.1 The 3D figure

"BO & Prediction" merged cell on the **Cross-Trajectory Predictors** slide (deck slide with `\slidenote{14}`):
- Used at `writing/presentation/deck/slides/06_xdecpfn.tex:209`: `\esbomerged{2}{0}` (view 2 = "wall view", style 0 = nested band at T only).
- Defined in `writing/presentation/deck/slides/00_local_macros.tex:389-544`: `\mergedtrials` (`:406`), analytic functions `g, m, s, mu, sd, crash` (`:408-415`), `\mergedband` fills (`:416-420`), `\mergedcell{view}{style}` (`:421-510`, 3 views via TikZ `x/y/z` unit vectors at `:423-430`), `\esbomerged` wrapping it with the Experiment and Early-stopping-rule panels (`:514-544`).
- Snapshot copies: `writing/presentation/clean/full/macros.tex:277,334` and `clean/{full,static}/slides/06_cross.tex` (fixed to the chosen view, no args). Origin and iteration log: `writing/presentation/variants/cross_trajectory_panels/NOTES.md:6-46`.
- **Technology:** pure TikZ (no pgfplots), pseudo-3D by oblique projection. Compiled with pdflatex in the dsme beamer theme.

What it shows: axes `t`, `J_t`, `θ`. Three blue trajectories = **real rig cost curves** J_t, each normalised by its own final cost and
scaled to the schematic predicted total cost m(θ), cut at its stop time (A ran to T, B and C early-stopped, with dotted drop lines).
Orange nested fills = predicted total cost μ±σ, μ±2σ at t=T over θ. Orange star + dashed guide = next controller θ_n. Light gray
grid on wall/floor/side. Red flat curve above the wall = predicted crash probability over θ.
Style rule (memory "3D: only blue lines", Tony 2026-10-11): inside the 3D space only the blue trajectories are lines, uncertainty as fills.

Inputs:
- `writing/presentation/shared/figures/furuta_traj/furuta_traj.tex` (generated) ← `writing/presentation/tools/make_furuta_traj_suggestion.py` (stdlib only). `MERGED` at `:79`: A = `plain_gp/seed4` trial 25 (stop 2.6 = full), B = `plain_gp/seed1` trial 12 (stop 0.9), C = `plain_gp/seed0` trial 3 (stop 0.6), all from `impl/prediction/hwplay/runs_holdcost5_newbox_reset_fixed/lqr_hardware` (`:26`).
- Everything else (m, s, crash) is hand-shaped schematic, not model output.

Turning it into a README graphic:
1. **Cheapest, exact look:** a `standalone` TikZ document that `\input`s the macros + `furuta_traj.tex`, compiled with pdflatex, converted with `dvisvgm --pdf` (present on the Mac at `/Library/TeX/texbin/dvisvgm`) or `pdftocairo -svg` (`/opt/homebrew/bin/pdftocairo`). Needs the RWTH color definitions from `shared/template/beamerthemedsme.sty:20-29` and the `pnl *` TikZ styles.
2. **Reproducible from repo data:** a ~150-line matplotlib `mplot3d` script: `Poly3DCollection` for the bands, `ax.plot3D` for the three rig trajectories read from `trials.jsonl`, same m/s/crash formulas, `ax.view_init` tuned to the wall view, axes panes off. Optionally animate (GIF) rotating view or growing trajectories. Plotly is not worth the dependency.
3. Hybrid: generate the coordinates in Python, emit TikZ, as the current tool already does.

### 5.2 Style conventions to unify

Thesis matplotlib (`impl/prediction/util/mpl_styles.py`, wrapped by `impl/dynformer/thesis_figures/style.py`, 1,706 lines):
- `formal_style`: `text.usetex=True`, Charter via `\usepackage[charter]{mathdesign}\usepackage[scaled=1.0]{zlmtt}`, `font.size 9`, serif, white bg (`mpl_styles.py:23-80`). Mathtext fallback when `OPTIM_DISABLE_USETEX=1` or `mathdesign.sty` missing (`style.py:49-60`).
- Figure sizes (inches): full width `(6.3, 2.0)`, 2-up `(3.02, 2.0)`, 3-up `(2.02, 2.0)`, tall `(6.3, 2.6)` (`mpl_styles.py:23-25`, `style.py:126-133`). Panel geometry constants `TARGET_BOX_PT 98.3`, `PANEL_GAP_PT 61.5`, `ROW_SPAN_PT 396.7` (`style.py:283-307`).
- Grid: dashed, lw 0.5, `#B0B0B0`, alpha 0.8 (`style.py:249`). Text halo via translucent bbox, never `path_effects` under usetex (`ENVIRONMENT.md:237-239`).
- Base colors: blue `#00549f`, green `#57AB27`, red `#CC071E` (= RWTH, `mpl_styles.py:36-38`). Crash marker = red cross (`style.py:147`).
- **One color per method family** (`style.py:1573-1630`): DecPFN `#0072B2`, LC-PFN `#D55E00`, FT-PFN `#009E73`, XDecPFN `#CC79A7`, TabPFN `#9B4FA8`, Segment-TabPFN `#882255` (rate mode `#C0506A`), Segment-GP/ESBO-GP `#E69F00`, ESRS `#BBBBBB`, BO `#8C564B`, DecPFN-H `#0B3C73`, Oracle `#000000`. Deadtime vs no-deadtime `#0072B2` / `#56B4E9`. Mostly Okabe-Ito.
- Canonical display-name map lives in `style.py` (`DISPLAY_NAME`, `OPT_FAMILY_COLORS`, `opt_family`).

Presentation (`writing/presentation/CLAUDE.md:68-92`, `docs/rwth_colors.md`):
- dsme beamer theme, RWTH palette: `RWTHBlue #00549F`, `RWTHOrange #F6A800`, `RWTHRed #CC071E`, `RWTHGreen #57AB27`, Bordeaux `#A11035` (`shared/template/beamerthemedsme.sty:20-29`). Sans text, serif math (`preamble.tex:42`).
- In schematics: trajectories/context = RWTHBlue, query/prediction = RWTHOrange (`00_local_macros.tex` near the end, `pfnctx`/`pfnqry`), crash = RWTHRed.
- Writing rules: AmE, no semicolons, "early stopping" unhyphenated (also apply to README text).

Note the two palettes differ: thesis method colors (Okabe-Ito) vs slide schematic colors (RWTH). A shared `style.py` in the new repo
should expose both: `METHOD_COLORS` for result plots and `RWTH` for schematics/README art.

---

## 6. Existing tests

No pytest suite, no `conftest.py`, no `[tool.pytest]`. Tests are standalone scripts that print/write a report.

| file(s) | what | runnable on Mac? |
|---|---|---|
| `impl/prediction/hwplay/selfcheck.py` | runs one cell 4× (clean, interrupted, killed, link failure) on the in-process simulator and demands bit-identical runs | yes in principle (sim only, torch CPU). Needs `PYTHONPATH=.:util` |
| `impl/dynformer/experiments/c1r2/c1r2_{data,train,bo,grid,gen,ladder,score,final,transfer,transfer_gen,transfer_train}_tests.py` | XDecPFN data path, calibration, model wrapper on a fake tranche (`c1r2_fake.py`). "no HPCWORK, no generator, no GPU" (`c1r2_data_tests.py:1-10`) | mostly yes (CPU, seconds). `c1r2_gen_tests` may need the C++ binaries |
| `impl/dynformer/predplay/predplay/{c1r2_group,c1r2_prefix,recipe_group}_tests.py` | predplay adapters. Legacy snapshot comparisons, skip missing checkpoints (`recipe_group_tests.py:149,290`) | partly. Checkpoint-dependent parts skip |
| `impl/dynformer/experiments/recipe/driver/test_{families,gpus}.py` | checkpoint discovery on fake dirs, GPU picker on fake `nvidia-smi` output | yes |
| `impl/dynformer/experiments/{s11,s12,f2,f31}/*_tests.py` | exploration-phase law/grid tests. `s12_data_tests.py` needs `$HPCWORK/s12/pool` + a GPU UUID | mostly no (pools) |
| `impl/dynformer/experiments/landscape_sampler/test_{a,b,d}.py` | statistical comparisons that read `out/*.npz` | no (data) |
| `impl/dynformer/experiments/i10h/i10h_g1_selfcheck.py` | phase-specific | unclear |
| `impl/prediction/tools/{furuta,cartpole}_cpp/validate.py` (+ `.slurm`) | C++ generator vs Python simulator equivalence | needs the built binary (Linux/MKL today) |
| `impl/prediction/training/lcpfn/predict_test.py` | legacy LC-PFN script | not a test |
| `impl/furuta/tests/` | only an mp4 | — |
| `impl/pfns4bo_test/*.py`, `impl/furuta/bad_scripts/test_method*.py` | scratch | no |

Good seeds for a new `tests/`: hwplay selfcheck (resumability), C++ vs Python validate, recipe driver tests, c1r2 data tests.

---

## 7. Git

- **One repo** at `bachelorarbeit/` (no nested `.git` anywhere, `find -name .git`). Remote `origin https://github.com/tonyborchert/bachelorarbeit.git`. Branches `master-new` (current, `origin/HEAD`), `main`, `backup-before-filter-repo` (a `git filter-repo` has been done before).
- `.git` is 1.47 GB (packs 1.40 GB). 6,675 tracked files, ~2.17 GB working-tree size of tracked files.
- `bachelorarbeit/new/` currently sits **inside** this repo (untracked). To become its own repository it should be moved out, or initialised as a fresh repo and ignored by the parent (`/new/` in root `.gitignore`). A fresh `git init` (no history import) is the clean choice given the history size and the private material below.
- `.gitignore`s: root (`/.uv/`, `.aider*`, `__pycache__/`, `writing/progress/presentation.key`), `impl/.gitignore` (236 lines, GitHub Python template + `out/`, `*.ckpt`, `*.ckpt.lock`, `*.pt`, `*.raw_npy`, `*.npy` at `:230-236`), `impl/prediction/.gitignore` (10), `impl/dynformer/.gitignore` (21: LaTeX aux, `logs/`, `state/`, some `out/`), `impl/furuta/.gitignore` (`*.so`, builds), `writing/.gitignore`, `writing/presentation/.gitignore` (`out/`).
- Largest tracked files (do not carry over blindly): `impl/prediction/notebooks/optimization/optimizer_plotter.ipynb` 99 MB, `notebooks/prediction/robust2_decpfn_lcpfn.ipynb` 67 MB, `impl/prediction/ESBO-main/results/simulation_experiment.mat` 67 MB, `handoff_matched_live_prediction/checkpoints/matched_comparison_boiler_40phi.tar.gz` 57 MB, `impl/furuta/bad_scripts/recording.mp4` 48 MB, presentation rig videos 12–26 MB each, `impl/prediction/hwplay/results/trials_lqr_hardware*.csv` 11/7.5 MB, **prebuilt Linux ELF `impl/prediction/tools/furuta_cpp/furuta_gen_phi` 9.8 MB**, `dynformer/experiments/predplay_decpfn_gap/A_ceilings/data/*.parquet` 15 MB each.
- Must not go public: `writing/thesis/sources/*.pdf` (copyrighted papers, 15–21 MB each), `writing/inspiration/*`, `writing/thesis/critique/*`, `writing/midterm/feedback.pdf`. `impl/prediction/ESBO-main/` looks like a vendored copy of Stenger et al.'s code (check its license).
- Raw rig data that the thesis needs is small enough to version or to attach as a release asset: the two thesis trees are 121 MB + 114 MB (JSONL, compresses well).

---

## 8. Local machine (Mac)

Measured on this machine:

| tool | status |
|---|---|
| `python3` | `/usr/bin/python3` 3.9.6 (system, too old). uv-managed CPython 3.12.13 and 3.14.6 in `bachelorarbeit/.uv/python/` |
| `uv` | 0.12.1 (Homebrew) |
| C/C++ | Apple clang 21.0.0 (`/usr/bin/clang++`, `g++` is the clang shim), `make` |
| `cmake`, `gfortran` | not installed |
| BLAS/LAPACK via brew | none (`brew list` has no openblas/lapack/mkl/eigen/libomp). Apple **Accelerate** framework is always available |
| TeX | MacTeX: `pdflatex`, `latexmk`, `dvisvgm` in `/Library/TeX/texbin` |
| Poppler | `pdftoppm`, `pdftocairo` |
| HW | Apple M5, 10 cores, 24 GB, arm64 |

C++ generators on macOS:
- `cartpole_cpp`, `plants_cpp`: plain C++17 + pthread. Fail only on `ARCH ?= x86-64-v3` (invalid for arm64 clang) and `-static-libstdc++ -static-libgcc` (unsupported by Apple ld). With `make ARCH=native LDFLAGS="-lpthread -lm"` they should build. Not verified (read-only audit).
- `furuta_cpp`: `Makefile:17-21` hard-errors without `MKLROOT`. `linalg.hpp:38-60` declares Fortran LAPACK symbols `dgebal_, dgges_, dtgsen_, dgetrf/dgesv, dgeqrf/dorgqr, dgesdd` (LP64). **No LAPACK fallback exists.** Accelerate exports exactly these classic symbols (deprecated, but present), so a macOS branch `LDFLAGS = -framework Accelerate` (or `-lopenblas` on Linux without MKL) is a small Makefile change. Numerics may differ from MKL in the last bits, which matters only for byte-identical pool reproduction (validate against scipy, which also uses Accelerate/OpenBLAS on Mac).
- No Linux-only syscalls found in the generator sources (grep for `/proc`, `sched_setaffinity`, `fallocate`, `posix_fadvise`, `MAP_POPULATE`: none).
- The tracked `furuta_gen_phi` is a Linux x86-64 ELF. Never ship binaries, build them.

Torch on the Mac: `impl/prediction/.venv` already has `torch 2.12.1` (MPS wheel), `tabpfn 8.1.0`, `botorch 0.18.1`, `gpytorch 1.15.2`.
CPU-only (or MPS) training of tiny PFNs for smoke tests is viable. TabPFN prefers MPS automatically (`tpfn/__init__.py:70-76`).
A full `uv sync` on the Mac currently fails because of the missing `gym_brt` path dependency (§1.7).

---

## 9. Open questions for Tony

1. **Output root.** One env var (e.g. `ESBO_PFN_HOME`) with subdirs `data/`, `models/`, `runs/optplay`, `runs/predplay`, `runs/hwplay`, defaulting to `./outputs` locally and `$HPCWORK/<project>` on the cluster? Or keep separate `DATA`/`MODELS`/`OPTPLAY_CHECKPOINTS`/`PREDPLAY_DATA`/`HWPLAY_RUNS` overrides?
2. **HPC specifics.** OK to move accounts, partitions, module lines and GPU-selection policy into a `cluster/rwth_claix.toml` profile, with a `local` profile as default? Should the agent budget gate (`slurm_submit.sh`) survive, or be dropped from the public repo?
3. **Login-node GPU training.** The thesis FT-PFN/XDecPFN models were trained on login-node H100s with resume-after-30-min loops. Should the new repo document that path, or only a clean Slurm `c23g` job (and accept queue time)?
4. **Hardware driver.** Vendor `gym_brt` (BlueRiverTech, which license?) plus `remote_env_server.py` from the workstation's `impl/furuta/furuta/quanser-driver`, or keep the rig as an optional extra that documents the websocket protocol only? The HIL SDK itself cannot be redistributed. Can you copy `quanser-driver/` (with `remote_env_server.py` and `setup.py`) from the workstation, since it is not on the Mac or in git?
5. **Which rig data ships.** Only the two thesis trees (`…newbox_reset_fixed`, `…_different_plant`, ~235 MB raw), or also `runs_holdcost5_newbox` (source of the reset-replay benchmark) and the earlier fields? In git, Git LFS, or as a Zenodo/GitHub-release archive?
6. **Checkpoints.** Where do the final checkpoints live publicly (TabPFN v3 cannot be redistributed, likely fetched by the user, but DecPFN/LC-PFN/XDecPFN/FT-PFN/rigreset DecPFNs are yours)? Hugging Face, Zenodo, or GitHub release? Copy them now from `$HPCWORK`/`/work` before the allocation (`thes2287` ends 2026-11-10) expires.
7. **`/work/dln18500` paths.** The Furuta XDecPFN and FT-PFN checkpoints are on `$WORK`, not `$HPCWORK`. Confirm they still exist and should be collected with the rest.
8. **C++ on macOS/Linux without MKL.** OK to add an Accelerate/OpenBLAS fallback to `furuta_cpp`, accepting last-bit differences to the MKL pools used in the thesis?
9. **Python version.** Pin the new repo to 3.12 (HPC module 3.12.3), or allow 3.13? (`<3.14` is current, furuta's Mac venv was 3.14 but is irrelevant.)
10. **3D README figure.** TikZ→SVG (exact slide look, needs TeX in the build) or a matplotlib re-rendering from the rig JSONL (pure Python, slightly different look)?
11. **Excluded material.** Confirm dropping `impl/cartpole_real`, `impl/pfns4bo_test`, `impl/test_project`, `impl/furuta/bad_scripts`, and the vendored `ESBO-main/` (or keep it as a cited baseline if its license allows).
