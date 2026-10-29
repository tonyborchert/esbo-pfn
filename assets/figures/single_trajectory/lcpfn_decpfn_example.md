# lcpfn_decpfn_example

Thesis asset: `writing/thesis/assets_generated/simulation/prediction/lcpfn_decpfn/lcpfn_decpfn_example.pdf` (fig:sim:lcpfn-decpfn-example)

Files: `lcpfn_decpfn_example.png`

## What is shown
- Six flush panels on one Cartpole trial: LC-PFN (left triple) and DecPFN (right triple), 1M binned, each at cutoffs 35, 65 and 85 % of T (snapped to the model's last predictable step at or before the cutoff).
- Models: `lcpfn/cartpole/thesis-binned-linearnorm/1M`, `decpfn/cartpole/thesis-binned/1M`. Context: the first 64 trials of the run (context-blind models ignore it).
- Black: the true cumulative cost J_t over the whole trial, dashed: the cutoff, blue strip right of T: predictive density of J_T conditioned on J_T >= J_t, dark blue bar: its mean. Corner text: the predicted crash probability.

## Metric
- No ticks. x = step t, y = J_t, one shared y limit 9844.1 (1.6 J_T or the trajectory maximum).

## Key numbers
- LC-PFN 35%: cutoff step 350, n = 64, crash 47.1%, mean 6904.6, std 1070.8, observed 1613.4
- LC-PFN 65%: cutoff step 650, n = 64, crash 19.2%, mean 6295.1, std 282.2, observed 3462.0
- LC-PFN 85%: cutoff step 850, n = 64, crash 4.4%, mean 6336.3, std 151.8, observed 5170.6
- DecPFN 35%: cutoff step 350, n = 64, crash 6.4%, mean 6299.8, std 519.4, observed 1613.4
- DecPFN 65%: cutoff step 650, n = 64, crash 4.6%, mean 6258.0, std 378.3, observed 3462.0
- DecPFN 85%: cutoff step 850, n = 64, crash 0.0%, mean 6170.6, std 614.0, observed 5170.6

## Data
- Run i0008_s00, query trial 64, T = 1000, true J_T = 6152.58.
- query named by the imported thesis task table `tasks-thesis`: run i0008_s00, trial index 64, thesis J_T = 6649.86
- Oracle run simulated live on instance 8 of the thesis-imported instance list (seed 0 of the `fields` kind), cached under `$PFN_HOME/figures/_cache/single_trajectory/`
- Live results (simulated runs, true slice, model predictions) are cached as npz under `$PFN_HOME/figures/_cache/single_trajectory/`, keyed by plant, slice, model checkpoint hash and context. Delete that directory to recompute them.
