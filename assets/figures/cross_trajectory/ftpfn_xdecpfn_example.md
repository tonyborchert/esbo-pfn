# ftpfn_xdecpfn_example

Thesis asset: `writing/thesis/assets_generated/simulation/prediction/ftpfn_xdecpfn/ftpfn_xdecpfn_example.pdf` (fig:sim:ftpfn-xdecpfn-example)

Files: `ftpfn_xdecpfn_example.png`

## What is shown
- Six flush panels on the Cartpole trial of `lcpfn_decpfn_example`: FT-PFN (left triple) and XDecPFN (right triple), all cut at 35 % of T (snapped per model), columns = context size n = 0, 5, 10.
- Models: `ftpfn/cartpole/thesis/default`, `xdecpfn/cartpole/thesis/default`.
- Context: trials of the same run drawn uniformly without the query, nested (one permutation with seed 20260920, sizes are prefixes), each given in full length. Drawn trials: n=0: []; n=5: [3, 12, 13, 15, 47]; n=10: [3, 8, 12, 13, 15, 23, 32, 39, 47, 60].
- Black: true J_t, dashed: cutoff, blue strip: predictive density of J_T conditioned on J_T >= J_t, dark blue bar: its mean. Corner text: crash probability and context size.

## Metric
- No ticks. x = step t, y = J_t, shared y limit 9844.1.

## Key numbers
- FT-PFN n=0: cutoff step 350, n = 0, crash 18.6%, mean 6074.2, std 309.6, observed 1613.4
- FT-PFN n=5: cutoff step 350, n = 5, crash 1.0%, mean 6158.4, std 303.0, observed 1613.4
- FT-PFN n=10: cutoff step 350, n = 10, crash 0.2%, mean 6033.3, std 197.2, observed 1613.4
- XDecPFN n=0: cutoff step 350, n = 0, crash 0.3%, mean 5976.9, std 251.8, observed 1613.4
- XDecPFN n=5: cutoff step 350, n = 5, crash 0.1%, mean 6023.8, std 276.9, observed 1613.4
- XDecPFN n=10: cutoff step 350, n = 10, crash 0.1%, mean 5997.5, std 232.5, observed 1613.4

## Data
- Run i0008_s00, query trial 64, T = 1000, true J_T = 6152.58.
- query named by the imported thesis task table `tasks-thesis`: run i0008_s00, trial index 64, thesis J_T = 6649.86
- Oracle run simulated live on instance 8 of the thesis-imported instance list (seed 0 of the `fields` kind), cached under `$PFN_HOME/figures/_cache/cross_trajectory/`
- Live results (simulated runs, true slice, model predictions) are cached as npz under `$PFN_HOME/figures/_cache/cross_trajectory/`, keyed by plant, slice, model checkpoint hash and context. Delete that directory to recompute them.
