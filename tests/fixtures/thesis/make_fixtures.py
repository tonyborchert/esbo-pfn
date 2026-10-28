"""Write the small fixture files of the thesis importer tests (run once, outputs are committed).

The files follow the schemas of the old result files: a fig1 table, a context-ladder parquet with
its sidecar, a timing bundle with two records, an optplay instance list with one instance and one
raw optplay run file (``OptimizerBenchmark`` checkpoint schema) with a crashed, an early-stopped and
a complete trial.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main() -> None:
    pd.DataFrame({
        "model_id": ["dynformer_fr2", "dynformer_fr2", "tabpfn/resolution50/typels"],
        "group": ["dynformer_fr2", "dynformer_fr2", "tabpfn"],
        "n_ctx": [0, 2, 2], "y": [0.17, 1.09, 0.5], "n": [1250, 1250, 1250],
        "n_problems": [5, 5, 5], "clamped": [0.0, 0.0, 0.0], "oos": [0.0, 0.0, 0.0],
        "y_min": [0.17, 1.09, 0.5], "y_max": [0.17, 1.09, 0.5],
    }).to_csv(HERE / "fig1_cross.csv", index=False)

    rows = []
    for run_id in ("run_phi0000_s00", "run_phi0003_s01"):
        for n in (0, 2):
            for frac, step, ll in ((0.1, 125, -8.0), (0.5, 500, -6.0)):
                rows.append({"p_fail": 0.01, "pred_mean": 100.0, "pred_std": 10.0,
                             "pred_median": 100.0, "ll": ll, "ll_conditioned": ll + n * 0.5,
                             "cond_mass": 1.0, "clamped": False, "out_of_support": False,
                             "trunc_skipped": False, "problem": "cartpole", "run_id": run_id,
                             "n_ctx": n, "trial_index": 180, "cutoff_frac": frac,
                             "cutoff_step": step, "model_id": "dynformer_fr2",
                             "group": "dynformer_fr2", "params_json": "{}", "target_J": 105.0})
    pd.DataFrame(rows).to_parquet(HERE / "cartpole__dynformer_fr2.parquet", index=False)
    (HERE / "cartpole__dynformer_fr2.json").write_text(json.dumps(
        {"problem": "cartpole", "model_id": "dynformer_fr2", "query_index": 180,
         "contexts": [2, 0], "k_selected": 2}, indent=1))

    base = {"cutoff_frac": 0.5, "n_calls": 1, "measured_at": "2026-09-15T17:51:14+00:00",
            "host": "artemis", "device": "cuda:Quadro RTX 6000", "torch_threads": 1,
            "cpu": "x86_64", "torch": "2.14.0+cu126"}
    records = [
        {**base, "model_id": "decpfn/epochs100/headbinned", "group": "decpfn",
         "problem": "boiler", "n_ctx": 0, "seconds_median": 0.002, "seconds_mean": 0.002,
         "seconds_p90": 0.002, "context_blind": True},
        {**base, "model_id": "dynformer_c1r2/cellf_s20260931", "group": "dynformer_c1r2",
         "problem": "furuta_lqr_rigreset", "n_ctx": 64, "seconds_median": 0.05,
         "seconds_mean": 0.05, "seconds_p90": 0.06},
    ]
    bundle = HERE / "timing_bundle"
    bundle.mkdir(exist_ok=True)
    (bundle / "predplay_timing.json").write_text(json.dumps({"records": records}, indent=1))

    inst = {"problem": "boiler", "dataset": "boiler/universal", "d": 1, "T": 701,
            "screen": {"n_theta": 100, "n_reps": 10, "min_pass": 25},
            "instances": [{"index": 0, "phi": [1.869, 0.1402], "noise_stds": [0.0221],
                           "phi_hat": None, "attempt": 0,
                           "screen": {"accepted": True, "pass_count": 25, "theta_screened": 25,
                                      "trials_used": 250}, "rejected": []}]}
    (HERE / "instances").mkdir(exist_ok=True)
    (HERE / "instances" / "boiler.json").write_text(json.dumps(inst, indent=1))

    T = 501

    def trial(t: int, crash: bool, slope: float) -> dict:
        J = (np.arange(1, t + 1) * slope).tolist()
        failed = [False] * t
        if crash:
            failed[-1] = True
        return {"t": t, "T": T, "theta": [0.1, -0.2, 0.3], "phi": [], "failed": failed,
                "x": [], "X": [], "y": [], "Y": [], "j": [], "J": J}

    record = {"key": "k", "optimizer": "Plain GP", "phi_index": 3, "repetition": 0, "seed": 7,
              "trials": [trial(12, True, 1.0), trial(T, False, 2.0), trial(200, False, 3.0),
                         trial(T, False, 1.0)],
              "successful": [False, True, False, True], "elapsed": 9.5,
              "stop_reason": "t_budget", "optimizer_time": 4.0,
              "performance_times": {"fitting": 1.0, "acquisition": 2.0},
              "performance_warmup": False,
              "optimizer_early_stops": [{"trial_index": 2, "observed_t": 200, "natural_t": T,
                                         "would_succeed": True, "full_cost": 1503.0,
                                         "incumbent": 1002.0}],
              "early_stop_diagnostics_complete": True, "workers": 1}
    runs = HERE / "optplay" / "pt2_deadtime@g0.05@bcapped" / "phi_0003" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "Plain_GP.json").write_text(json.dumps(
        {"version": 3, "dataset": "pt2_deadtime/universal", "optimizer": "Plain GP",
         "runs": [record]}))


if __name__ == "__main__":
    main()
