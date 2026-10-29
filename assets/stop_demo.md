# stop_demo

## What is shown
- animated replay of logged rig trials: pendulum drawn from the logged angles, cumulative cost (blue) against the best final cost so far (dashed), the forecast density of the final cost (orange below, red above the best so far), Pr[final cost >= best so far] with the stop level, the predicted crash probability; after a stop the rest of the logged trial is revealed (dotted)

## Metric
- forecast = `predict` of the running prefix with all earlier trials of the run as context, updated at the decision steps
- Pr[worse] = survival at the best so far, crash mass included (the `naive` rule of `xdecpfn-pi`)

## Data
- `hardware_optimization` units of the rig and the checkpoint named below

## Numbers
- rig `without_weight`, method `bo` seed 0 (provenance thesis), model `xdecpfn/furuta/thesis/default`, rules `exceed` + `naive` at tau = 0.99, forecasts at step 0 and at the decoder grid steps
- trial 9: best so far 279.03, not stopped, logged J_T 267.91 at step 600, Pr[worse] at step 0 0.215
- trial 11: best so far 267.91, stopped by `forecast` at step 189 (1.575 s), Pr[worse] 0.9928, p_crash 0.000, logged J_T 334.60 at step 600, Pr[worse] at step 0 0.844
- trial 19: best so far 265.45, stopped by `forecast` at step 126 (1.050 s), Pr[worse] 0.9940, p_crash 0.697, crashed at step 197 (1.642 s), Pr[worse] at step 0 0.791
