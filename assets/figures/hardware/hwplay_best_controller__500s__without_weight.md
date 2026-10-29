# hwplay_best_controller__500s__without_weight

## What is shown
- rig `without_weight` (without weight), methods ['xdecpfn-pi', 'esbo-gp']
- left: arm angle, right: pendulum angle (dashed, unwrapped, 0 deg upright, the trace shifted by a multiple of 360 deg so that it ends upright), degrees over time within the trial (120 Hz)

## Metric
- controller = the incumbent (best successful trial so far) at 500 s of reset-normalized wall-clock of the seed with the median incumbent cost (index (n - 1) // 2 of the ranked seeds), one rig execution, traces from the stored x channels of that trial

## Data
- rig `without_weight`: 30 `hardware_optimization` units (provenance `thesis`) in `$PFN_HOME/benchmarks/db.sqlite`, methods and seeds bo [0, 1, 2, 3, 4], decpfn [0, 1, 2, 3, 4], decpfn-h [0, 1, 2, 3, 4], esbo-gp [0, 1, 2, 3, 4], esrs [0, 1, 2, 3, 4], xdecpfn-pi [0, 1, 2, 3, 4]

## Numbers
- xdecpfn-pi: median seed 2, incumbent trial index 20, J = 258.1, kick sign +1, theta [74.878, 1.075, 12.072, 4.779, 1.398, 0.665, 1.481, 2.924], incumbent cost per seed at 500 s (ranked) seed 1: 257.2, seed 0: 257.7, seed 2: 258.1, seed 3: 260.8, seed 4: 263.7
- esbo-gp: median seed 3, incumbent trial index 29, J = 263.6, kick sign -1, theta [85.962, 1.001, 12.184, 5.416, 0.978, 1.153, 1.381, 1.638], incumbent cost per seed at 500 s (ranked) seed 1: 260.6, seed 2: 262.2, seed 3: 263.6, seed 4: 263.7, seed 0: 267.6
