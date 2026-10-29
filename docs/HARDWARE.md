# Hardware: the Quanser QUBE-Servo 2 Furuta rig

Everything that touches the rig lives in `esbopfn.hardware`:

| module | role |
|---|---|
| `protocol.py` | the JSON messages between client and server |
| `client.py` | `RigClient`: `reset_down`, `step`, `center`, `begin_trial`, `close`, `connection_reset` |
| `server/` | `python -m esbopfn.hardware.server --plant {qube,sim}` |
| `sim_plant.py` | stepwise simulated rig on the `problems.furuta` internals |
| `reset.py` | the between-trial reset (thesis Algorithm alg:app:hw-reset) |
| `system.py` | `RigSystem`: one trial of the Furuta controller on the rig |
| `runner.py` | the rig field (`pfn hw run`), resumable |
| `records.py`, `store.py` | trial and run records, reading runs from the database |
| `metrics.py` | the quantities the hardware figures plot |
| `prediction_bench.py` | the hardware prediction benchmark and its ablation |
| `rigs.py` + `configs/rigs.yaml` | the open registry of rig configurations |

Benchmark kinds: `hardware_optimization` (one rig run) and `hardware_prediction` (one predictor on
one rig). The importer of the thesis logs is `esbopfn.benchmarks.import_hardware`.

## Setup on the workstation

The rig is driven by one Linux workstation with the QUBE on USB.

1. Install the Quanser HIL SDK (`/opt/quanser/hil_sdk`, headers and `libhil`,
   `libquanser_runtime`, `libquanser_common`).
2. Build BlueRiverTech's `gym_brt` (`quanser-openai-driver`) against it, in the Python environment
   that runs the server. Its Cython extension skips itself silently when the SDK is missing, so
   check `python -c "from gym_brt.quanser import QubeHardware"`.
3. Install this package in the same environment (`uv sync` or `pip install -e .`).
4. Start one server, and only one (the board admits one process, a second server fights over port
   8765 and the board):

   ```bash
   pgrep -af esbopfn.hardware.server        # must be empty
   python -u -m esbopfn.hardware.server --plant qube --port 8765
   ```

   `Board could not be opened` means another process holds the board. A wedged board (every
   channel bit-constant) needs a USB reset or a replug.
5. Opening the board zeroes the encoders where the arm stands (no homing). Center the arm by
   hand before the first start.

The `qube` backend uses the documented `gym_brt` API: `QubeHardware(frequency=120,
max_voltage=18)` as a context manager that opens the board, `reset_down()`, and
`step(np.array([v]))` returning `[theta, alpha, theta_dot, alpha_dot]` with `alpha` measured
from upright. Without `gym_brt` the server refuses `--plant qube` with a pointer to this file.

The `sim` backend needs nothing beyond this package:

```bash
python -m esbopfn.hardware.server --plant sim --port 8765 [--instance-json plant.json] [--realtime]
```

`--instance-json` takes `{"phi": {...}, "phi_hat": {...}, "noise_std": {...}, "extras":
{"dead_time": ...}}` (default: the furuta prior's nominal plant without noise). The simulated rig
uses the furuta simulator's dynamics, integrator, dead-time line and constants unchanged
(`problems/furuta/sim.py` has no stepwise entry point, so `sim_plant.py` is a stepper over its
`Dynamics`, `step` and `DeadTime`). It quantizes both angles to 2048 counts per revolution and
re-zeroes the arm encoder on `reset_down`. On `begin_trial(seed)` it restarts from a seeded rest
state (hanging, 0.01 residual), which makes a simulated field reproducible trial by trial
(`--keep-state` turns this off). `--realtime` paces it at 120 Hz, otherwise it steps as fast as
the client asks.

Server safety nets: voltages are clipped to `--max-voltage` (6 V, the controller's own limit),
more than `--max-motor-steps` (3600) consecutive steps with a nonzero voltage are refused with
0 V written, and a dropped connection releases the board.

## Protocol

One JSON object per websocket message, strictly request and reply.

| request | reply |
|---|---|
| `{"cmd": "hello"}` | `{"type": "hello", "plant": "qube"\|"sim", "frequency": 120.0, "protocol": 1, "reseed_state": bool}` |
| `{"cmd": "reset_down"}` | `{"type": "state", "state": [phi_arm, phi_pend, phidot_arm, phidot_pend]}` |
| `{"cmd": "step", "voltage": v}` | `state` after one control period at `v` volts |
| `{"cmd": "center", "u_max": 1.0, "max_seconds": 15.0, "tol": 0.02}` | `state` (bounded PD drive of the arm to the board zero, ends at 0 V) |
| `{"cmd": "begin_trial", "seed": s}` | `{"type": "ok"}` (seeds the simulated rig, ignored by the QUBE) |
| `{"cmd": "close"}` | `{"type": "closed"}` (0 V, board released) |

Errors come back as `{"type": "error", "message": "..."}` and raise `RigError` in the client.
Angles are in rad, `phi_arm` in the board frame, `phi_pend` from upright in `(-pi, pi]`
(hanging is `+-pi`).

The client waits at most 15 s for a `step` reply, 90 s for `reset_down`, 60 s for `center` and
30 s for `close`. On a timeout it discards the socket (a late reply must not answer the next
request), raises `RigTimeout`, and reconnects on the next command. Keepalive pings are off
(`ping_interval=None`) because the server cannot answer them while it blocks in a driver call.

## The reset between trials

`RigReset.run(rng, reopen)` executes the thesis algorithm step by step, with every duration
counted in 120 Hz control periods:

1. Brake the pendulum until `|phidot_pend| < 0.2` rad/s and `||phi_pend| - pi| <= 3` deg over a
   trailing 0.4 s window (at most 8 s). Brake law: `-0.3 min(|phidot_pend|, 6) sign(-cos(phi_pend)
   phidot_pend)` V above a 0.05 rad/s deadband, plus `0.8 (phi_arm -+ 20 deg)` V once the arm is
   more than 20 deg out, clipped to 2 V, off beyond 80 deg of arm travel.
2. Coast at 0 V for 1.5 s. If all four channels stayed bit-constant (a frozen board), reset the
   connection, which reopens the board, and check that it reads again.
3. PD drive of the arm to the session center, `u = 3.0 phi_arm + 0.2 phidot_arm` V clipped to
   1.5 V, at least 0.35 V while the arm is stuck (`|phidot_arm| < 0.05` rad/s), until
   `|phi_arm| < 2` deg with `|phidot_arm| <= 0.3` rad/s for 0.1 s, 3 s elapsed, or the arm moves
   more than 8 deg further away from the center. Coast 0.3 s.
4. Repeat: recenter (step 3) if `|phi_arm| > 5` deg, brake until the rig was still for 0.5 s,
   coast 0.3 s, hold 0 V, until for 2 s continuously `||phi_pend| - pi| < 2` deg,
   `|phidot_pend| < 0.05` and `|phidot_arm| < 0.05` rad/s. After 25 s the trial starts anyway and
   the diagnostics record `settled: false`.
5. The current arm position becomes the zero of `phi_arm` for the trial.
6. Draw `s` from the trial's generator (`substream(seed, n)`) and apply `s * 0.6` V for 0.03 s
   (4 periods), then one 0 V step, and the trial starts.

Every tenth trial (`n` a positive multiple of 10) the connection is reset after step 1: the arm
is centered and settled first, its position is read, the connection is closed and reopened and
`reset_down` reopens the board, and the read position is carried across the reopen so the
session's arm frame survives it.

Diagnostics recorded per trial (`reset` in the trial record): `kind` (`reset`, `initial`,
`connection_reset`), `brake_stopped`, `frozen`, `center_before_deg`, `center_after_deg`,
`center_converged`, `center_reached`, `center_aborted`, `center_floor_steps`, `settled`,
`settle_attempts`, `settle_recenters`, `settle_dwell_restarts`, `settle_longest_dwell_s`,
`settle_arm_deg`, `settle_hang_deg`, `settle_arm_dot`, `settle_pend_dot`, the `reopen_*` keys on
connection resets, `kick_sign`, `kick_voltage`, `zero_deg`, `arm_after_deg`, `pend_dot`,
`phase_s` and `phase_steps` per phase, `steps`, `seconds`.

## One trial

`RigSystem.run_trial` runs the furuta controller of `problems.furuta.sim` (energy swing-up, arm
barrier, LQR designed from the rig's nominal `p_hat`, the same for every rig configuration) for
up to 600 steps. At step `k` the measured state is logged, its hold cost
`|phi_pend| (1 + 20 (1 - min(|phi_pend|, 5 deg) / 5 deg))` is added, the early-stop callback sees
the trial cut at `k` (for `k < 600`) and may end it before the next command is sent, then the
command goes out and the trial crashes at `k` if the unwrapped arm angle leaves `+-90 deg`.
Success: complete and mean `|phi_pend|` over the last 2 s below 5 deg.

## Running a rig field

```bash
pfn hw list                                  # rigs and field defaults
pfn hw run --rig without_weight              # six methods x seeds 0-4, N_max 180, T_budget 36000
pfn hw run --rig with_weight --methods decpfn,decpfn-h --seeds 0-4 --url ws://localhost:8765
pfn hw status --rig with_weight
pfn hw pause                                 # stop after the current trial (pfn hw pause --clear)
pfn hw run --rig without_weight --plant sim  # the whole field against an in-process simulated rig
```

Methods: `esrs, bo, esbo-gp, decpfn, decpfn-h, xdecpfn-pi` with the strategies of
`strategies.make_strategy`. `decpfn` and `decpfn-h` run at `tau = 0.9` with
`decpfn/furuta/binned/1M`, `xdecpfn-pi` at `tau = 0.99` with `xdecpfn/furuta/default/default`
(`configs/rigs.yaml`, `field.method_params`). Seeds are shared across methods, so every method
starts from the same initial design (`substream(seed, "init")`, `N_init = 8` uniform trials).

The budgets, initial design, incumbent and `exceed` rule are those of `strategies.loop`. The rig
loop cannot reuse `run_optimization` directly, because that loop simulates a trial in full and
replays the stop rules afterwards, so `runner.py` re-implements the loop around an online trial
and imports the shared pieces (`incumbent_index`, `sample_uniform`, `strategy_rng`, the seed
streams). The next controller is proposed before the reset starts and the two never overlap.

Each unit writes `run.json` and appends one line per trial to `trials.jsonl` in its blob
directory `<PFN_HOME>/benchmarks/blobs/hardware_optimization/<key hash>/`. A rerun replays the
stored trials into a fresh strategy (it proposes again and checks that the stored controller comes
out, streams the stored prefixes through the stop rules, ends each trial) and continues on the
rig. A rig error rebuilds the unit the same way and retries. On the simulated rig a resumed field
is identical to an uninterrupted one in everything except wall-clock times.

## Adding a rig configuration

Add an entry under `rigs:` in `src/esbopfn/configs/rigs.yaml` with a `description`, a
`mass_note`, the `reset_reference_method` (`decpfn-h`) and `j_best: best_on_rig`. Never edit an
existing entry for a physically different rig: runs are keyed by the rig name, and pooling runs of
two plants would mix their regret references.

## Records

`hardware_optimization` key: `{"rig", "method", "params", "seed", "n_max", "t_budget",
"provenance"}`. Trial line (`records.HwTrial`): `index, theta, source, T, t_obs, outcome
(complete | early_stopped | crashed | budget_cut), crash_step, success, J_obs, J, x, xdot, u,
kick_sign, incumbent_index, incumbent_J, timings {propose_s, reset_s, control_s, stop_rule_s,
end_s, total_s}, reset {...}, max_arm_deg, mean_pend_final_deg, started_at, extra`.

## Metrics of the figures

`metrics.py`: reset-normalized wall-clock (every trial's reset time replaced by the mean reset
time of `decpfn-h` on that rig), `J_best` (best successful cost on the rig across the declared
methods), median log10 regret with 25/75 % seed bands over wall-clock and over the first 60
trials, per-seed regret at 50 and 100 % of the wall-clock budget with the seed that defines
`J_best` excluded, mean crash count over wall-clock, median best successful raw cost over the
first 1000 s, the incumbent at 500 s of the median seed with its arm angle and unwrapped
pendulum angle, and the mean acquisition wait per trial.

## Prediction benchmark

```bash
pfn hw bench-pred --rig without_weight                       # Segment-GP, DecPFN, XDecPFN
pfn hw bench-pred --rig with_weight --ablation               # the four DecPFN models
pfn hw bench-pred --rig with_weight --models decpfn=decpfn/furuta/binned/1M --logs native
```

Queries are trials that were not stopped early (complete or crashed) and are not the first of
their run. At context size `n` in `0, 1, 2, 5, 11, 22, 45, 90` a query is scored only if it has
`n` predecessors and gets exactly those. Cutoffs are 10/30/50/70/90 % of the query's own length,
snapped to the closest step at or before them where the predictor is defined (XDecPFN drops
cutoffs before its first grid point). The score is the conditioned log-likelihood, averaged over
a query's cutoffs, and the summary holds the median per context size over non-crashed queries
(`median_ll`) and over all queries (`median_ll_all`), with the median inference time. The
ablation gives every query its full history and scores DecPFN with and without the dead time in
its prior and with and without state channels (`configs/benchmarks/hardware.yaml`).

`hardware_prediction` key: `{"rig", "bench", "predictor", "model", "ladder", "cutoff_fracs",
"methods", "logs", "provenance"}`. Blob: `scores.parquet` (columns `rig, method, seed,
query_index, n_ctx, n_ctx_used, cutoff_frac, cutoff_step, step, predictor, ll, p_crash, crashed,
target_J, query_t, infer_s`, `n_ctx = -1` for the full history) and `summary.json`.

## Importing the thesis logs

```bash
pfn hw import-logs <thesis project>/impl/prediction/hwplay/runs_holdcost5_newbox_reset_fixed
pfn hw import-logs <thesis project>/impl/prediction/hwplay/runs_holdcost5_newbox_reset_fixed_different_plant
pfn hw import-pred <thesis project>/impl/dynformer/experiments/hwpred/results
```

or from Python `import_rig_logs(root, rig_name=None, home=None)` and
`import_hwpred_results(results_dir, home=None)` (`esbopfn.benchmarks.import_hardware`), or
`import_all(impl_root)`. The tree `runs_holdcost5_newbox_reset_fixed` is the rig without weight,
`..._different_plant` the rig with weight. Cells map to methods as `random_stenger` -> `esrs`,
`plain_gp` -> `bo`, `stenger` -> `esbo-gp`, `mtr0.99_k06binned` -> `decpfn`,
`htmr0.99_k06binned` -> `decpfn-h`, `qc_pi_0.99` -> `xdecpfn-pi`. The cell `qc_pi_0.99_n10k`
is not imported. Imported units carry `provenance: "thesis"` and the old cells' parameters
verbatim in `params` (the DecPFN cells ran the matched rule at 0.99 with a crash threshold of
0.9). The field-by-field conversion is documented in `import_hardware.py`.

The old prediction benchmark counted the 14 queries of `qc_pi_0.99_n10k` on the rig without
weight (757 queries at `n = 0`). They are dropped here, so the imported and native query sets
of that rig have 743. Its old ladder medians were taken over all queries, the ablation medians
over the non-crashed ones (`median_ll_all` and `median_ll` in the summaries).

## Untested without the rig

The unit tests cover the protocol, the client against an in-process simulated server, the
reset (completes, diagnostics, connection reset), full trials, the early-stop callback, the
field runner with an interrupted and resumed run, the metrics, the prediction benchmark and the
importers. Not tested here: the `qube` backend against `gym_brt` and the HIL SDK (written
against the documented driver API), real-time behavior at 120 Hz with the driver blocking in HIL
calls, the reset's tolerances on the physical arm (stiction, self-centering, the frozen-board
path), USB re-enumeration and board wedges. `tests/hardware/test_qube_roundtrip.py` is the
first check on the workstation (`uv run pytest tests/hardware --run-hardware` with the qube
server running).
